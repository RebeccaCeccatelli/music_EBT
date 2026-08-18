#!/bin/bash
# Monitor job 13304472 and convert MIDI to WAV when done

JOB_ID=13339571
CONDA_ENV="/home/rebcecca/.conda/envs/music_EBT/bin/python"

echo "Waiting for job $JOB_ID to complete..."

while true; do
    STATE=$(sacct -j $JOB_ID --format=State --noheader 2>/dev/null | head -1 | xargs)

    if [[ "$STATE" == "COMPLETED" ]] || [[ "$STATE" == "FAILED" ]]; then
        echo "✅ Job $JOB_ID $STATE"
        break
    fi

    echo "  Status: $STATE (waiting...)"
    sleep 10
done

# Run conversion
echo ""
echo "Converting MIDI to WAV..."
$CONDA_ENV convert_midi_simple.py

echo ""
echo "✅ All done! Files ready at:"
echo "  GPT2:  inference_outputs/gpt2_comparison/sample_*_generated.wav"
echo "  Llama: inference_outputs/llama_comparison/sample_*_generated.wav"
