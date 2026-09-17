# One-click local setup

The launcher supports **Windows 10/11 x64**. No existing Python, Node, Git, HF
login or API key is needed. First launch needs internet access to GitHub,
nodejs.org, PyPI, the PyTorch wheel server and Hugging Face.

## Choose a version

| Version | What it prepares | Launch from a Git clone |
|---|---|---|
| Starter | Main17 tasks, saved rankings, fusion inputs and local thumbnails | `start.bat` |
| Full | Everything in Starter, plus the selected dataset's embeddings, patch tokens and pretrained probe banks | `start-full.bat` |

Download [Starter](https://github.com/lyfdy456/ProbeScout/releases/download/v0.2.0-launcher/ProbeScout-Starter-Windows.zip)
or [Full](https://github.com/lyfdy456/ProbeScout/releases/download/v0.2.0-launcher/ProbeScout-Full-Windows.zip).
Extract the ZIP before running `start.bat`; each package preselects its edition.
Both download the chosen assets on first launch. A Git clone includes both entry
points, and the setup page also lets you change editions.

Starter supports the paper's **Weight Tune / Staged** with frozen probes and
gates. Full adds assets for further training and the optional **Update Probes**
extension. It installs CPU PyTorch; GPU training uses the separate
[manual environment](manual_setup.md#1-install-the-code-dependencies).

## Open the workspace

1. Download and extract your [original dataset images](manual_setup.md#3-place-original-images-for-image-galleries).
2. Double-click the launcher. On first use it prepares a local Python runtime,
   then opens the setup page in your browser.
3. Select Cars, HICO-DET or CelebA. Click **Browse** to select the image folder,
   or paste its path. Images can stay outside the repository.
4. Click **Prepare & open**. The page shows installation and download progress.
   When thumbnails are ready, ProbeScout opens automatically.

For Cars, select the folder containing `000001.jpg` through `016185.jpg` from
`car_ims`. For HICO-DET, select `images/`, containing `train2015/` and `test2015/`.
For CelebA, select `img_celeba/` from **In-The-Wild Images**.

Use the same launcher next time. It remembers the image folder and reuses assets.
The setup page remains available to change the dataset, upgrade to Full or retry
a failed installation. One dataset is enabled at a time; previously downloaded
datasets remain on disk. Keep the launcher window open while using the Web app.
Press **Ctrl+C** or close that window to stop its local services.

## Download and disk sizes

| Dataset | Starter task download | Full assets, including tasks | Suggested free space for Full |
|---|---:|---:|---:|
| Cars | 94 MB | 7.16 GB | 12 GB |
| HICO-DET | 320 MB | 17.38 GB | 23 GB |
| CelebA | 439 MB | 63.11 GB | 130 GB during installation |

Sizes exclude original images. Allow roughly 5 GB for a Starter installation,
including runtimes, packages, task files and thumbnails. CelebA's patch array is
automatically reconstructed from shards; the shards are retained, so this step
needs another 60.99 GB. Full downloads no additional original images. A later
operation that encodes attribute text can download the SigLIP backbone once.

## Local files and retries

Environment files, download progress and image paths live in `.probescout/`.
Tasks/features are placed under `dataset/`, and probe banks under
`visual_analytics/pcp_analyze/runtime/`. These directories are excluded from Git.
The launcher uses pinned HF revisions and checks SHA-256 before installing assets.
Interrupted downloads resume where supported; completed files are reused.

If setup fails, correct the folder or network problem and click **Retry**.
The page shows the error and recent logs; the full log is `.probescout/launcher.log`.
Missing images are reported before large Full assets are downloaded. Use the
correct dataset variant if the query-image checksum differs.

The launcher selects free loopback ports and opens the actual Web address.
Manual `npm run dev` keeps its usual ports 3000 and 8787. New-task creation is
still a [command-line workflow](new_tasks.md).
