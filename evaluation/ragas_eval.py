import json

from datasets import Dataset


def read_json(path):
    with open(path, "r") as f:
        return json.load(f)
    

def get_rag_answer():
    return "Pretend I answered"


def get_eval_dataset(path):
    """
    Load and prepare the evaluation dataset.
    """

    data = read_json(path)["data"]

    questions_list = []
    ground_truths_list = []
    contexts_list = []
    answers_list = []

    for text in data:
        for paragraph in text["paragraphs"]:
            for qa in paragraph["qas"]:
                question = qa["question"]
                ground_truth = qa["answers"][0]["text"]
                context = paragraph["context"]
                answer = get_rag_answer()

                questions_list.append(question)
                ground_truths_list.append(ground_truth)
                contexts_list.append(context)
                answers_list.append(answer)

    data_dict = {
        "question": questions_list,
        "answer": answers_list,
        "contexts": contexts_list,
        "ground_truths": ground_truths_list,
    }
    
    dataset = Dataset.from_dict(data_dict)

    return dataset


def main():
    """"""
    data_path = "evaluation/data/test.json"
    dataset = get_eval_dataset(data_path)
    print(dataset)


if __name__ == "__main__":
    main()