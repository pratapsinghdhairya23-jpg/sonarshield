# Model Checkpoint

Place a trained YOLO-Seg-compatible checkpoint here as:

```
models/sonarshield_yoloseg.pt
```

If this file is present AND `torch` is installed AND it loads successfully,
`ai/inference.py` will report:

    Inference Mode: YOLO-Seg

Otherwise the application automatically falls back to the Prototype
heuristic engine (`ai/prototype_engine.py`) and reports:

    Inference Mode: Prototype

## Wiring up real inference

`ai/yolo_seg_engine.py`'s `YOLOSegInferenceEngine.run()` currently raises
`NotImplementedError` on purpose — this repository does not fabricate a
forward pass. Once you have a real trained checkpoint and know its export
format (raw `torch.save`, TorchScript, ONNX, or an Ultralytics `.pt`),
implement `run()` to:

1. Preprocess the input the same way your training pipeline expects.
2. Run the forward pass.
3. Postprocess into the same detection dict shape used by
   `ai/prototype_engine.run_prototype_detection()`:
   `{ "bbox": [x, y, w, h], "mask": <bool ndarray>, "features": BlobFeatures(...), "detection_confidence": float }`

No accuracy numbers, training claims, or dataset-trained status should be
displayed anywhere in the UI unless they are verifiably true for the
checkpoint actually loaded.
