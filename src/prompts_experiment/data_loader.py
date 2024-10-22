import json

def load_gsm_ic_data(dataset):
    """
    dataset: '2step' (34220), 'mstep' (23832), 'validation' (1000)
    """
    problems = []

    if dataset == '2step':
        fp = '/home/cpp/jerryhuang/reasoning/data/gsm_ic/GSM-IC_2step.json'
    elif dataset == 'mstep':
        fp = '/home/cpp/jerryhuang/reasoning/data/gsm_ic/GSM-IC_mstep.json'
    else:
        fp = '/home/cpp/jerryhuang/reasoning/data/gsm_ic/gsm8k_validation.jsonl'

    if 'jsonl' in fp:
        with open(fp, "r") as file:
            for line in file:
                line = line.strip()
                if line:  # Skip empty lines
                    try:
                        problems.append(json.loads(line))
                    except json.JSONDecodeError:
                        print(f"Skipping invalid line: {line}")
    else:
        print('Loading a .json file')
        with open(fp, "r") as file:
            problems = json.load(file)

    return problems

if __name__ == "__main__":
    problems = load_gsm_ic_data('mstep')
    print(f"Loaded {len(problems)} problems")
    
    print(problems[0])
    
"""
{
   "original_question":"Steve is 5\\'6\".  He grows 6 inches.  How tall is he in inches?",
   "answer":"72",
   "new_question":"Steve is 5\\'6\". He grows 6 inches. The height of Emma is 8 feet. How tall is Steve in inches?",
   "n_steps":2,
   "role":"Emma",
   "number":"8",
   "sentence_template":"The height of {role} is {number} feet.",
   "role_label":"nonoverlapped",
   "number_label":"in_range",
   "sentence_l
"""