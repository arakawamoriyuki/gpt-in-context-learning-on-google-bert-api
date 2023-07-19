import os
import sys

from flask import Flask, jsonify, request
import requests
from dotenv import load_dotenv
import openai

from datasets import get_datasets


app = Flask(__name__)


@app.route('/questions', methods=['POST'])
def post_questions():
    messages = request.json['messages']
    last_message = messages[-1]
    text = last_message['content']

    res = requests.post(
        'http://localhost:8501/v1/models/app:predict',
        json={'inputs': [text]},
    )
    outputs = res.json()['outputs'][0]
    datasets_yml_path = './input/datasets.yml'
    _, _, _, _, answers = get_datasets(datasets_yml_path)
    prediction_zips = dict(zip(answers, outputs))
    sorted_predictions = sorted(prediction_zips.items(), key=lambda x: x[1], reverse=True)
    predictions = sorted_predictions[0:10]
    hint_text = '\n'.join([prediction[0] for prediction in predictions])

    context_messages = [
        {
            'role': 'system',
            'content': (
                'あなたはコールセンターの優秀なオペレータです。'
                '次のドキュメントの内容に従って返答してください。'
                '返答できない場合はお問い合わせに誘導してください。'
            ),
        },
        {
            'role': 'system',
            'content': hint_text,
        }
    ]
    load_dotenv()
    openai.api_key = os.getenv('OPENAI_API_KEY')
    res = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=context_messages + messages,
    )

    return jsonify({
        'message': res['choices'][0]['message']['content'],
        # 'context': predictions,
    }), 200


def main():
    app.run(port=8000, debug=True)


if __name__ == '__main__':
    main()
    sys.exit(0)
