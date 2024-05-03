import os
import argparse
import json
import re

from llava.eval.m4c_evaluator import TextVQAAccuracyEvaluator


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation-file', type=str)
    parser.add_argument('--result-file', type=str)
    parser.add_argument('--result-dir', type=str)
    return parser.parse_args()


def prompt_processor(prompt):
    if prompt.startswith('OCR tokens: '):
        pattern = r"Question: (.*?) Short answer:"
        match = re.search(pattern, prompt, re.DOTALL)
        question = match.group(1)
    elif 'Reference OCR token: ' in prompt and len(prompt.split('\n')) == 3:
        if prompt.startswith('Reference OCR token:'):
            question = prompt.split('\n')[1]
        else:
            question = prompt.split('\n')[0]
    elif len(prompt.split('\n')) == 2:
        question = prompt.split('\n')[0]
    else:
        assert False

    return question.lower()


def eval_single(annotation_file, result_file):
    experiment_name = os.path.splitext(os.path.basename(result_file))[0]
    print(experiment_name)
    annotations = json.load(open(annotation_file))['data']
    annotations = {(annotation['image_id'], annotation['question'].lower()): annotation for annotation in annotations}
    results = [json.loads(line) for line in open(result_file)]

    pred_list = []
    index = 0
    annotation_list = list(annotations.items())
    for result in results:
    # Method One (works universally, either out-of-order or in-order, but slow)
        #question = result['prompt'] # result is a Python Dictionary variable. "prompt should have the question in it"
        #question_lower = question.lower()  # Convert the question to lowercase, although it is already lowercase.
        # search through the annotations, and if we find a corresponding question, retrieve the annotation.
        #matching_annotation = 0
        #for key, annotation in annotations.items(): # slow bc it is looping through all of annotations... refer to the Method Two
        #    if question == key[1]:  # Check if the question matches
        #        matching_annotation = annotation
        #        break
        #if matching_annotation == 0:
        #    assert False # question was not found in the annotations
        #pred_list.append({
        #    "pred_answer": result['text'],
        #    "gt_answers": matching_annotation['answers'],
        #})
    # Method Two (only works if the result files are in-order, but faster)
        matching_annotation = annotation_list[index][1]
        pred_list.append({
            "pred_answer": result['text'],
            "gt_answers": matching_annotation['answers'],
        })
        index += 1

    evaluator = TextVQAAccuracyEvaluator()
    print('Samples: {}\nAccuracy: {:.2f}%\n'.format(len(pred_list), 100. * evaluator.eval_pred_list(pred_list)))


if __name__ == "__main__":
    args = get_args()

    if args.result_file is not None:
        eval_single(args.annotation_file, args.result_file)

    if args.result_dir is not None:
        for result_file in sorted(os.listdir(args.result_dir)):
            if not result_file.endswith('.jsonl'):
                print(f'Skipping {result_file}')
                continue
            eval_single(args.annotation_file, os.path.join(args.result_dir, result_file))
