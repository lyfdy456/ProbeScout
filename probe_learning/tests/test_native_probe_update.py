"""Synthetic native warm-start checks; no task data or production checkpoints."""
import contextlib
import hashlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.methods import native_probe_update as native
from src.methods.mlp_probe import MLP
from src.methods.attn_probe import AttnPoolMLPWithProj
from src.methods.metric_attention_probes import LearnedQueryAttentionProbe, TripletProjectionProbe
from src.methods.nnpu_probe import nnpu_loss
from src.methods.dcpu_probe import dcpu_loss
from src.methods.pu_ranking_probe import pu_ranking_loss


class NativeProbeUpdateTests(unittest.TestCase):
    def setUp(self):
        self.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory(prefix="native-probe-test-")
        self.root = Path(self.temp.name)
        rng = np.random.default_rng(81)
        self.X = rng.normal(size=(12, 8)).astype(np.float32)
        self.y = np.asarray([0, 1] * 6)
        self.Xv = rng.normal(size=(4, 8)).astype(np.float32)
        self.yv = np.asarray([0, 1, 0, 1])
        self.U = rng.normal(size=(8, 8)).astype(np.float32)
        self.config = native.NativeUpdateConfig(epochs=1, lr=1e-3, batch_size=12,
            batch_size_u=4, weight_decay=0, anchor_strength=0,
            kfold_epochs=1, kfold_folds=2, triplets_per_step=4)

    def tearDown(self):
        self.temp.cleanup()
        torch.set_num_threads(self.old_threads)

    def case(self, method):
        torch.manual_seed(101)
        if method == "triplet_loss":
            model = TripletProjectionProbe(8, projection_dim=4)
        elif method == "attention_pooling":
            model = LearnedQueryAttentionProbe(8)
        elif method == "attribute_conditioned_attention":
            model = AttnPoolMLPWithProj(8, 4)
        else:
            model = MLP(8)
        attention = method in native.ATTENTION_METHODS
        path = self.root / f"{method}.pt"
        torch.save({"state_dict":model.state_dict(), "model_class":type(model).__name__,
                    "kind":"attn" if attention else "mlp", "input_dim":8,
                    "patch_dim":8 if attention else None, "text_dim":4 if attention else None}, path)
        original = path.read_bytes()
        X = np.repeat(self.X[:, None, :], 3, axis=1) if attention else self.X
        Xv = np.repeat(self.Xv[:, None, :], 3, axis=1) if attention else self.Xv
        kwargs = {
            "sample_weights":np.asarray([1] * 10 + [8, 16], dtype=np.float32),
            "validation_data":(Xv,self.yv), "fit_ids":[f"fit-{i}" for i in range(12)],
            "validation_ids":[f"val-{i}" for i in range(4)],
            "expected_checkpoint_sha256":hashlib.sha256(original).hexdigest(),
            "blocked_ids":["test-0", "query-0"], "config":self.config, "seed":3,
        }
        if method in native.PU_METHODS:
            kwargs.update(X_unlabeled=self.U, unlabeled_ids=[f"u-{i}" for i in range(8)])
        if method == "attribute_conditioned_attention":
            kwargs["text_query"] = torch.ones(1,4)
        return path, original, model, X, kwargs

    def test_uniform_weights_preserve_native_pu_and_ranking_losses(self):
        scores = torch.tensor([-1., .2, .7, -.4], requires_grad=True)
        labels = torch.tensor([1., 0., 1., 0.])
        U = torch.tensor([-.5, .9, .1], requires_grad=True)
        positive = labels == 1
        for method, expected in (
            ("nnpu", nnpu_loss(scores[positive],scores[~positive],U,.4)),
            ("dcpu", dcpu_loss(scores[positive],scores[~positive],U,.4,.5)),
            ("pu_ranking", F.binary_cross_entropy_with_logits(scores,labels,pos_weight=torch.tensor(2.)) + pu_ranking_loss(scores[positive],U)),
        ):
            actual = native.native_labeled_objective(method,scores,labels,torch.ones(4),
                pos_weight=torch.tensor(2.),unlabeled_logits=U,pi=.4)
            torch.testing.assert_close(actual,expected,rtol=0,atol=0)
            if method in ("nnpu","dcpu"):
                weights=torch.tensor([8.,1.,1.,1.])
                weighted=native.native_labeled_objective(method,scores,labels,weights,
                    pos_weight=torch.tensor(2.),unlabeled_logits=U,pi=.4)
                excess=7*F.binary_cross_entropy_with_logits(scores[:1],labels[:1])/2
                torch.testing.assert_close(weighted-actual,excess)
                grad_base=torch.autograd.grad(actual,U,retain_graph=True)[0]
                grad_weighted=torch.autograd.grad(weighted,U,retain_graph=True)[0]
                torch.testing.assert_close(grad_base,grad_weighted,rtol=0,atol=0)

    def test_all_eight_warm_start_native_objectives_and_freeze_outputs(self):
        for method in native.NATIVE_METHODS:
            with self.subTest(method=method), patch.object(native,"_get_device",return_value=torch.device("cpu")), \
                 patch("src.methods.pu_probe._get_device",return_value=torch.device("cpu")), contextlib.redirect_stdout(io.StringIO()):
                path, original, _model, X, kwargs = self.case(method)
                updated,audit=native.update_native_probe(method,path,X,self.y,**kwargs)
                self.assertEqual(path.read_bytes(),original)
                self.assertEqual(audit["fitCount"],12)
                self.assertEqual(audit["valCount"],4)
                self.assertEqual(audit["selectedEpoch"],1)
                self.assertTrue(audit["warmStarted"] and audit["frozenForFusion"])
                self.assertGreater(audit["parameterDisplacementSquared"],0)
                self.assertEqual(audit["objective"],native.OBJECTIVES[method])
                self.assertFalse(any(p.requires_grad for p in updated.parameters()))
                self.assertTrue(all(p.grad is None for p in updated.parameters()))
                probabilities=native.score_native_probe(method,updated,X,
                    text_query=kwargs.get("text_query"),chunk_size=3)
                self.assertEqual(probabilities.shape,(12,))
                self.assertTrue(np.isfinite(probabilities).all())
                self.assertTrue(((0<=probabilities)&(probabilities<=1)).all())
                # Changing only external Val can affect selection, never the
                # one-epoch training updates or KFold fit-only confidence.
                kwargs["validation_data"]=(-kwargs["validation_data"][0],1-self.yv)
                again,_=native.update_native_probe(method,path,X,self.y,**kwargs)
                for key,value in updated.state_dict().items():
                    torch.testing.assert_close(value,again.state_dict()[key],rtol=0,atol=0)

    def test_loading_is_real_warm_start_not_random_initialization(self):
        path,original,model,X,kwargs=self.case("mlp_baseline")
        with patch.object(native,"_get_device",return_value=torch.device("cpu")), \
             patch.object(torch.optim.Adam,"step",return_value=None):
            updated,audit=native.update_native_probe("mlp_baseline",path,X,self.y,**kwargs)
        for key,value in model.state_dict().items():
            torch.testing.assert_close(value,updated.state_dict()[key],rtol=0,atol=0)
        self.assertEqual(audit["parameterDisplacementSquared"],0)
        self.assertEqual(path.read_bytes(),original)

    def test_missing_mismatched_checkpoint_and_leaking_ids_fail_closed(self):
        path,_,_,X,kwargs=self.case("mlp_baseline")
        with self.assertRaises(FileNotFoundError):
            native.update_native_probe("mlp_baseline",self.root/"missing.pt",X,self.y,**kwargs)
        with self.assertRaisesRegex(RuntimeError,"checksum"):
            native.update_native_probe("mlp_baseline",path,X,self.y,**{**kwargs,"expected_checkpoint_sha256":"0"*64})
        for changed in (
            {"fit_ids":["val-0",*[f"fit-{i}" for i in range(1,12)]]},
            {"fit_ids":["fit-0"]*12},
            {"fit_ids":["test-0",*[f"fit-{i}" for i in range(1,12)]]},
            {"sample_weights":np.zeros(12)},
            {"validation_data":None},
            {"X_unlabeled":self.U,"unlabeled_ids":["val-0",*[f"u-{i}" for i in range(1,8)]]},
        ):
            with self.subTest(changed=list(changed)),self.assertRaises(ValueError):
                native.update_native_probe("mlp_baseline",path,X,self.y,**{**kwargs,**changed})
        bad_y=self.y.copy(); bad_y[0]=2
        with self.assertRaises(ValueError):
            native.update_native_probe("mlp_baseline",path,X,bad_y,**kwargs)
        with self.assertRaisesRegex(RuntimeError,"projection|architecture"):
            native.update_native_probe("triplet_loss",path,X,self.y,**kwargs)

    def test_kfold_auxiliary_receives_only_merged_fit_and_base_hash_is_guarded(self):
        path,original,_,X,kwargs=self.case("kfold_pu")
        with patch.object(native,"_get_device",return_value=torch.device("cpu")), \
             patch.object(native,"cross_validate_probs",return_value=np.full(12,.5)) as oof:
            _,audit=native.update_native_probe("kfold_pu",path,X,self.y,**kwargs)
        np.testing.assert_array_equal(oof.call_args.args[1],X)
        np.testing.assert_array_equal(oof.call_args.args[2],self.y)
        self.assertEqual(oof.call_args.kwargs["epochs"],self.config.kfold_epochs)
        self.assertEqual(audit["kfoldPolicy"],"cold-init-fit-only-oof; warm-start-final-probe")
        path,original,_,X,kwargs=self.case("mlp_baseline")
        with patch.object(native,"_get_device",return_value=torch.device("cpu")), \
             patch.object(Path,"read_bytes",side_effect=[original,original+b"changed"]), \
             self.assertRaisesRegex(RuntimeError,"changed while training"):
            native.update_native_probe("mlp_baseline",path,X,self.y,**kwargs)

    def test_no_fusion_mode_argument_and_invalid_config_rejected(self):
        path,_,_,X,kwargs=self.case("mlp_baseline")
        with self.assertRaises(TypeError):
            native.update_native_probe("mlp_baseline",path,X,self.y,mode="joint",**kwargs)
        for config in (native.NativeUpdateConfig(epochs=0),native.NativeUpdateConfig(lr=float("nan")),
                       native.NativeUpdateConfig(kfold_folds=1),native.NativeUpdateConfig(anchor_strength=-1)):
            with self.assertRaises(ValueError): config.validate()


if __name__ == "__main__":
    unittest.main()
