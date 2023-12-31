import os
import sys

from dotenv import load_dotenv
import openai


def main():
    load_dotenv()
    openai.api_key = os.getenv('OPENAI_API_KEY')
    res = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        # model='gpt-4',
        messages=[
            {
                'role': 'system',
                'content': 'You are a helpful assistant.',
            },
            {
                'role': 'user',
                'content': 'Who won the world series in 2020?',
            },
            {
                'role': 'assistant',
                'content': (
                    'The Los Angeles Dodgers'
                    'won the World Series in 2020.'
                ),
            },
            {
                'role': 'user',
                'content': 'Where was it played?',
            },
        ],
    )

    print(res['choices'][0]['message']['content'])


if __name__ == '__main__':
    main()
    sys.exit(0)
