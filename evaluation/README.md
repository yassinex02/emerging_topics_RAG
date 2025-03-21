# Ragas evaluation

## Instructions to run the ragas evaluation:

If you have already previously downloaded the evaluation dataset `test.json`, you can skip ahead to step 4.

1. **Go to the evaluation data directory:**
```
cd evaluation/data
```

2. **Download the test data with:**
```
wget https://huggingface.co/datasets/PlanTL-GOB-ES/SQAC/resolve/main/test.json
```

3. **Go back to the root directory:**
```
cd ../..
```

4. **Add your OPENAI API KEY to a `.env` file:**  
The evaluation uses OpenAI models as evaluators.
```
OPENAI_API_KEY=your_api_key
```

5. **Run the ragas evaluation script:**  
You need to provide a mandatory argument (`--notes`), containing info about the experiment, such as the tei / tgi model used, or any other change.
```
python evaluation/ragas_eval.py --notes "Write your experiment notes here"
```

## Summary of experiments:  
*Insert markdown table containing the same info as the csv file basically*