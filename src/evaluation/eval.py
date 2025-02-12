


from lark import Lark
def check_relations(predicted, correct):
    """
        compute precision, recall, f1

        and score from 0 to 5 assigned this way:
            - 1 point if relation exists
            - up to 5 points for the cardinality
    """

    total_score = 0
    true_pos = 0
    tp_fn = len(correct)
    tp_fp = len(predicted)
    
    for rel in correct:
        score = 0
        for rel_ in predicted: 
            if set(rel.keys()) == set(rel_.keys()):
                true_pos += 1
                score += 1
                for r in rel.keys():
                    if ".." not in rel[r]:
                        rel[r] = '"' + rel[r].replace('"','') + ".."+ rel[r].replace('"','') +'"'

                    if ".." not in rel_[r]:
                        rel_[r] = '"' + rel_[r].replace('"','') + ".."+ rel_[r].replace('"','') + '"'
        
                    c_min = rel[r].split("..")[0]
                    c_min_ = rel_[r].split("..")[0]

                    if c_min == c_min_:
                        score+=1

                    c_max = rel[r].split("..")[1]
                    c_max_ = rel_[r].split("..")[1]


                    if c_max == c_max_:
                        score+=1

        total_score += score

    if true_pos > 0 :
        precision = true_pos / tp_fp 
        recall = true_pos / tp_fn 
        f1 = 2 * ((precision * recall) / (precision + recall))
    else:
        precision = 0
        recall = 0
        f1 = 0
    
    return {
        "total_score":total_score,
        "precision": round(precision,2),
        "recall": round(recall,2),
        "f1": round(f1,2),
        "len":len(correct) * 5
    }
            
def check_class(predicted, correct):
    """
    Computes precision, recall, and F1-score between the predicted and correct lists.

    Args:
        predicted (list): The list of predicted values.
        correct (list): The list of correct (ground truth) values.

    Returns:
        dict: A dictionary with precision, recall, and F1-score.
    """

    # precision: how many predicted that are in correct.
    if type(predicted) != type(set()):
        precision =  (len(set(predicted).intersection(set(correct)))) / len(set(predicted))
    else:
        precision =  (len((predicted).intersection((correct)))) / len((predicted))

    
    # recall : how many predicted are not in correct.
    if type(predicted) != type(set()):
        recall =  (len(set(predicted).intersection(set(correct)))) / len(set(correct)) 
    else:
        recall = (len((predicted).intersection((correct)))) / len((correct)) 

    # f1: classic f1
    f1 = 2 * ((precision * recall) / (precision + recall))

    return {
        "precision": round(precision,2),
        "recall": round(recall,2),
        "f1": round(f1,2)
    }

def init_parser():
    with open("/Users/marcocalamo/Downloads/grammar.ebnf", encoding="utf-8") as grammar_file:
        parser = Lark(grammar_file.read())
        return parser
        
def parse_text(parser, ref):
    ref = ref.replace('`','')
    ref = ref.replace('plantuml','')

    if not '@startuml' in ref:
        ref = '@startuml\n' + ref
    if not '@enduml' in ref:
        ref = ref+'\n@enduml'
    #ref = ref.replace('@startuml','')
    #ref = ref.replace('@enduml','')
    parsed_ref = parser.parse(ref)
    return parsed_ref

def get_from_parsed(parsed_text, to_get="class"):
    found = parsed_text.find_pred(lambda v: v.data.value == to_get)
    
    to_ret = []

    for f in found:
        if to_get == 'relationship':
            one_rel = f.children[0].children            
            first_rel = one_rel[0].children[0].value
            assert one_rel[0].children[0].type == 'CNAME'

            try:
                first_card = one_rel[1].children[0].value
                assert one_rel[1].children[0].type == 'ESCAPED_STRING'
            except:
                first_card = "0..*"

            try:
                second_card = one_rel[5].children[0].value
                assert one_rel[5].children[0].type == 'ESCAPED_STRING'
            except:
                second_card = "0..*"

            second_rel = one_rel[-2].children[0].value
            assert one_rel[-2].children[0].type == 'CNAME'
            
            rel_tuple = {
                first_rel:first_card,
                second_rel:second_card
            }
            
            to_ret.append(rel_tuple)
        elif to_get == 'class':
            to_ret.append(f.children[0].children[0].value)
    
    return to_ret
    
def parse_path(path_to_uml, parser):
    
    if os.path.isfile(path_to_uml):
        with open(path_to_uml) as plant_ref:
            
            ref = plant_ref.read()
            parsed_ref = parse_text(parser, ref)
            found_class = get_from_parsed(parsed_ref, to_get="class")
            found_rel = get_from_parsed(parsed_ref, to_get="relationship")

            return found_class, found_rel
    else:
        raise FileNotFoundError

def evaluate(path_uml, path_to_check, parser):

    # gold class and rel
    class_uml, rel_uml = parse_path(path_uml, parser)

    # candidate class and rel
    class_check, rel_check = parse_path(path_to_check, parser)

    # check class
    res_cl = check_class(class_check, class_uml)

    # check rel
    res_re = check_relations(rel_check, rel_uml)

    return res_cl, res_re

import os
import csv

def process_folders(root_folder):
    csv_file = os.path.join(root_folder, "evaluation_results.csv")
    header_written = False
    
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        for sub_folder in os.listdir(root_folder):
            sub_folder_path = os.path.join(root_folder, sub_folder)
            if not os.path.isdir(sub_folder_path):
                continue

            uml_file = os.path.join(sub_folder_path, "uml.txt")
            if not os.path.exists(uml_file):
                continue
            
            for filename in os.listdir(sub_folder_path):
                if filename.startswith("result_") and filename.endswith(".txt"):
                    result_file = os.path.join(sub_folder_path, filename)
                    model_name = "-".join(filename.replace(".txt", "").split("_")[1:])
                    parser = init_parser()
                    try:
                        results = evaluate(uml_file, result_file, parser)
                        if not header_written:
                            writer.writerow(["sub_folder_name", "model_name", 
                                             "precision_class", "recall_class", "f1_class",
                                             "precision_rel", "recall_rel", "f1_rel", "score_rel", "max_score"])
                            header_written = True
                        writer.writerow([sub_folder, model_name, results[0]["precision"], results[0]["recall"], results[0]["f1"],
                                        results[1]["precision"], results[1]["recall"], results[1]["f1"],
                                        results[1]["total_score"], results[1]["len"]])
                    except Exception as e:
                        print(e,sub_folder, model_name)
                        #raise Exception

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_data_class(csv):
    # Load the CSV file
    df = pd.read_csv(csv)
    
    # Compute average f1_class and f1_rel per model_name
    avg_metrics = df.groupby('model_name')[['f1_class', 'f1_rel']].mean().reset_index()
    
    # Set seaborn style
    sns.set_theme(style='whitegrid')
    
    # Plot average f1_class per model
    plt.figure(figsize=(10, 5))
    sns.barplot(x='model_name', y='f1_class', data=avg_metrics, palette='Blues', hue='model_name')
    plt.title('Average F1 Class per Model')
    plt.xlabel('Model Name')
    plt.ylabel('Average F1 Class')
    plt.xticks(rotation=45)
    plt.show()


def plot_data_rel(csv):
    df = pd.read_csv(csv)
    
    # Compute average f1_class and f1_rel per model_name
    avg_metrics = df.groupby('model_name')[['f1_class', 'f1_rel']].mean().reset_index()
    
    # Set seaborn style
    sns.set_theme(style='whitegrid')

    # Plot average f1_rel per model
    plt.figure(figsize=(10, 5))
    sns.barplot(x='model_name', y='f1_rel', data=avg_metrics, palette='Reds', hue='model_name')
    plt.title('Average F1 Rel per Model')
    plt.xlabel('Model Name')
    plt.ylabel('Average F1 Rel')
    plt.xticks(rotation=45)
    plt.show()


def plot_data_score(csv):
    df = pd.read_csv(csv)
    
    # Compute average f1_class and f1_rel per model_name
    avg_metrics = df.groupby('model_name')[['score_rel', 'max_score']].mean().reset_index()
    #avg_metrics = avg_metrics.melt()
    # Set seaborn style
    sns.set_theme(style='whitegrid')

    # Plot average f1_rel per model
    plt.figure(figsize=(10, 5))
    sns.barplot(x='model_name', y='score_rel', data=avg_metrics, palette='Greens', hue='max_score', legend=False)
    plt.title('Average score per model')
    plt.xlabel('Model Name')
    plt.ylabel('Average score Rel')
    plt.xticks(rotation=45)
    plt.show()