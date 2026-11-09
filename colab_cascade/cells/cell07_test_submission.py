all_dets = []
test_durations = {}
for sid in TEST_STREAM_IDS:                                        # test_00 ... test_04
    t0 = time.time()
    sd = preprocess_stream(sid)                                    # SVD clutter removal -> 12 bands -> IQR norm
    test_durations[sid] = sd.duration
    result = run_cascade(sd, stage1_model, stage2_model)           # Stage 1 proposals -> Stage 2 FPN decoder
    dets = detections_from_outputs(result, sid, sd.duration, score_floor=SOFT_NMS_FLOOR)   # decode + Soft-NMS
    all_dets += dets
    print(f"{sid}: {sd.duration / 60:6.1f} min | {len(result['regions']):3d} suspicious blocks "
          f"({sum(b - a for a, b in result['regions']) / sd.duration:4.0%} of stream) | "
          f"{len(dets):3d} falls | {time.time() - t0:.0f}s")
    del sd, result; gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

submission = pd.DataFrame(
    [{"stream_id": d["stream_id"], "t_start": round(d["start_time"], 3), "t_end": round(d["end_time"], 3),
      "score": round(d["score"], 6)} for d in all_dets],
    columns=["stream_id", "t_start", "t_end", "score"],
).sort_values(["stream_id", "t_start"]).reset_index(drop=True)
submission.to_csv(SUBMISSION_CSV, index=False)
print(f"\nSaved {len(submission)} detections -> {SUBMISSION_CSV}")

# ---- format check with the official validator (stream ids, t_start < t_end, inside the stream) ----
if abc_eval is not None:
    abc_eval.validate_predictions(SUBMISSION_CSV, test_durations)
    print("Official validate_predictions: OK")
submission.head(10)
