from flask import Flask, render_template, request
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertForSequenceClassification
import warnings

app = Flask(__name__)

# Load fine-tuned model and tokenizer
tokenizer = BertTokenizer.from_pretrained('./fine_tuned_bert')
model = BertForSequenceClassification.from_pretrained('./fine_tuned_bert')

# Tokenize text using fine-tuned model and tokenizer
def encode_text(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    return outputs.logits

# Evaluate clauses
def compare_clauses(ground_truths, llm_output):
    clauses = [clause.strip() for clause in llm_output.split('. ', ', but', ', and', ', or', ', nor', ', yet', ', so') if clause.strip()] # split clauses by conjunctions, periods
    results = []
    sum_lowest_similarities = 0

    for clause in clauses:
        lowest_similarity = float('inf')
        worst_relationship = None
        contradicted_truths = []

        # Check contrast to each truth once per clause
        for ground_truth in ground_truths:
            ground_truth_vec = encode_text(ground_truth)
            clause_vec = encode_text(clause)
            similarity = cosine_similarity(ground_truth_vec.detach().numpy(), clause_vec.detach().numpy())[0][0]
            relationship = classify_relationship(similarity)

            # Determine the lowest similar between this clause and any ground truth
            if similarity < lowest_similarity:
                lowest_similarity = similarity
                worst_relationship = relationship
            
            # Check if the clause contradicts the ground truth
            if relationship == 'Contradiction':
                contradicted_truths.append(ground_truth)
        
        sum_lowest_similarities += lowest_similarity
        results.append((clause, lowest_similarity, worst_relationship, contradicted_truths)) # append results for each clause

    accuracy = (sum_lowest_similarities / len(clauses)) * 100 if clauses else 0
    return results, accuracy

def classify_relationship(similarity):
    if similarity > 0.8:
        return 'Entailment'
    elif similarity < 0.4:
        return 'Contradiction'
    else:
        return 'Neutrality'

@app.route('/')
def index():
    return render_template('input.html')

@app.route('/detect', methods=['POST'])
def compare():
    ground_truths = request.form.getlist('ground_truth')
    llm_output = request.form['llm_output']
    results, accuracy = compare_clauses(ground_truths, llm_output)
    return render_template('result.html', results=results, llm_output=llm_output, ground_truths=ground_truths, accuracy=accuracy)

if __name__ == '__main__':
    app.run(debug=True)
