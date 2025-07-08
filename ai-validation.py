import ast
import os
import dotenv
import spacy
from openai import OpenAI
import json


class EventParser:
    def __init__(self, username: str):
        # Load the spaCy language model
        self.nlp = spacy.load('en_core_web_md')
        self.username = username
        self._user_path = self.validate_username()

        # Load environment variables (for OpenAI API key)
        dotenv.load_dotenv()
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


    def validate_username(self)-> str:
        user_dir = f"../data/{self.username}"
        if not os.path.exists(user_dir):
            raise FileNotFoundError(f"Directory for {self.username} not found")
        post_dir = f"{user_dir}/posts"
        if not os.path.exists(post_dir):
            raise FileNotFoundError(f"Post Directory for {self.username} not found")
        if os.listdir(post_dir):
            return user_dir
        else:
            raise FileNotFoundError(f"No posts found for {self.username}")


    def parse_post(self, post_path: str):
        # Load the post data
        try:
            with open(post_path, 'r') as file:
                post_data = json.load(file)

            # Extract the text from the post data
            post_text = post_data['Description']
            post_date = post_data['Date']

            if 'Parsed' in post_data:
                raise Exception("Post has already been parsed.")

            """Parse dates from the text using OpenAI's GPT-4 API."""

            # Send request to OpenAI API to extract dates
            completion = self.client.chat.completions.create(
                model="gpt-4o-mini",  # Use the correct model name
                messages=[
                    {
                        "role": "system",
                        "content": "I want you to return the iso date with time for a piece of text. "
                                   "There could be multiple dates and times listed, so if there are multiple, "
                                   "separate them with commas. I will also give a context date in iso format, and if a "
                                   "day of the week is given, use the context date. Simply output a list of tuples in the format "
                                   "(\"iso date\", \"what the event is with any links provided\")."
                    },
                    {
                        "role": "user",
                        "content": f"{post_text} context date: {post_date}"
                    }
                ],
                temperature=0.5  # Adjust the creativity level
            )

            # Return parsed date results
            print("Successful parse.")
            return ast.literal_eval(completion.choices[0].message.content)

        except AttributeError as e:
            print(f"File cannot find correct attributes {e}")
            return []
        except Exception as e:
            print(f"Error while parsing: {e}")
            return []

    def parse_all_posts(self):
        post_dir = f"{self._user_path}/posts"
        for post in os.listdir(post_dir):
            post_path = f"{post_dir}/{post}"
            parsed_info = self.parse_post(post_path)
            self.store_parsed_info(post_path, parsed_info)

    def store_parsed_info(self, post_path: str, parsed_info: list):
        try:
            with open(post_path, 'r') as file:
                post_data = json.load(file)
            if 'Parsed' in post_data:
                raise Exception("Post has already been parsed.")
            post_data['Parsed'] = parsed_info
            with open(post_path, 'w') as file:
                print("Successfully stored")
                json.dump(post_data, file)
        except Exception as e:
            print(f"Error while storing parsed info: {e}")



if __name__ == "__main__":
    # Example usage
    parser = EventParser(username="wsg")
    parser.parse_all_posts()
