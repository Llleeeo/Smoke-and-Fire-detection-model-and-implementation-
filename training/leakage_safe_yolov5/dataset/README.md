# Clean dataset placeholder

The generated images and labels are intentionally not committed to GitHub. They contain 1,887 cleaned image/label pairs and should be rebuilt locally from the original YOLO export:

```bash
source ../.venv/bin/activate
python ../scripts/rebuild_clean_dataset.py /path/to/original-yolo-export /path/to/new-output-package
```

The committed `data.yaml` describes the expected `train`, `valid`, and `test` directory layout. Audit manifests and aggregate results are available in `../audit/`.
