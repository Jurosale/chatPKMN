import os
from random import sample

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from getkey import getkey, keys
from langchain_community.document_loaders import TextLoader, WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import requests

import create_pokemon_data as PKMN_DATA


SPECIAL_CHARS_IN_NAMES = [".", " ", "-"]
ENDING_PHRASE = "Thank you for talking to me!"
MAX_SUGGESTIONS = 10

user_pokemon_data = {} # Need to make this global so parser class can access it


class CustomParser(BaseOutputParser):
    """
    This parser object allows users to format a chat model's response.
    """
    def parse(self, text: str) -> str:
        # Assume pokemon speaks normally unless speech value says otherwise
        formatted_text = text
        
        # If pokemon speaks telepathically, showcase it
        if user_pokemon_data[PKMN_DATA.CSV_SPEECH_KEY] == "telepathy":
            formatted_text = "*Speaking telepathically* " + formatted_text

        # Else if pokemon speaks through pokemon noises, showcase it
        elif user_pokemon_data[PKMN_DATA.CSV_SPEECH_KEY] == "noise":
            formatted_text = "*" + user_pokemon_data[PKMN_DATA.CSV_NAME_KEY] + " noises* (" + formatted_text + ")"

        # Format should always be: [pokemon_name]([pokedex_id]): speech
        return "\n" + user_pokemon_data[PKMN_DATA.CSV_NAME_KEY] + " (#" + str(user_pokemon_data[PKMN_DATA.CSV_ID_KEY]) + "): " + formatted_text


# Perfroms a DFS of the trie to retrieve every full string name
# it can find based on the provided prefix string
def get_suggestions(input: str, trie_dict: dict) -> list[str]:
    # Use a recrusive search for DFS
    def _recursive_search(curr_str: str, curr_trie_dict: dict, suggest: list[str]):
        for next_char in curr_trie_dict.keys():
            if next_char == PKMN_DATA.END_NAME_STR:
                suggest.append(curr_str)
            _recursive_search(curr_str+next_char, curr_trie_dict[next_char], suggest)

    suggestions = []
    curr_dict = trie_dict
    for char in input:
        # If current prefix does not exist in trie,
        # then there aren't any names to suggest
        if char not in curr_dict:
            return suggestions
        curr_dict = curr_dict[char]
    _recursive_search(input, curr_dict, suggestions)

    # If there are more available suggestions than the max allowed,
    # randomly pick suggestions until the max amount is reached
    if len(suggestions) > MAX_SUGGESTIONS:
        suggestions = sample(suggestions, MAX_SUGGESTIONS)
    return suggestions


if __name__ == "__main__":
    # Print this statement to buy some time while setting up data & model
    print("Contacting PokeDex...")

    # Load up hidden API key located in a different file and set up chatbot model
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    chat_model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=api_key)
    
    # Retrieve pokemon data from websites
    pkmn_obj = PKMN_DATA.PokemonData()

    # Assign prompt & input
    template = """
        You are {pokemon_name}, a {type} type pokemon from generation {gen}. You are talking to a user who wants to ask you questions about yourself or give you requests. Follow the conversation rules listed below when responding to the user.

        Conversation Rules:
        1. Keep your responses to 3 sentences max unless the user explicitly gives you a response length to follow.
        2. If the user's input is inappropriate, reply with "I don't feel comfortable with this conversation. Would you like to talk about something else instead?".
        3. If the user's input is incomprehensible or empty, reply with "I don't understand. Would you like to ask me something else?".
        4. If the user asks something that is not related to the pokemon world or you in any way, reply with "I don't know anything outside of the pokemon world. Would you like to ask me something else?".
        5. If the user asks you something and you don't know the answer, reply with "I don't know. Would you like to ask me something else?".
        6. Do not lie or make up answers. Use the information found in "Important Information" as additional context (if possible) when deciding on a response.
        7. Use the conversation history in "Chat History" as additional context when evaluating the user's input.
        8. If the conversation ends, the user declines to ask you more questions or "Current Response Count" is greater than 10, do not ask a question and append "{end_phrase}" to the end of your response.
        
        Current Response Count:
        {count}
        
        Important Information:
        {context}
        
        Chat History:
        {chat_history}
    """
    human_template = "{text}"

    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    print("\nPokeDex: Hello, I am a virtual pokedex. I can help you talk to your favorite pokemon!")

    # Main loop where user chooses a pokemon to have a conversation with
    is_running = True
    while is_running:
        # 1st part of loop: Reset these chat model and helper variables
        user_pokemon_data = {}
        web_paths = [PKMN_DATA.PKMN_DB_WEB_PATH]
        chat_history = []
        user_input = ""
        retriever = ""
        result = ""
        key = ""
        response_count = 0

        # 2nd part of loop: Ensure either a pokemon is selected to converse with or user is done
        print("\nPokeDex: Please type in the pokemon's name or pokedex number to get started, press the tab key for a "
            + "suggestion based on current input or press the escape key to exit.")
        # Use getKey() instead of input() to both offer suggestions as the user types a name and filter out invalid chars
        # ESC key = exit program, BACKSPACE key = delete newest char, TAB key = give suggestion(s), ENTER key = accept user input
        while key != keys.ENTER:
            print(f"  -> {user_input}", end='\r') # This will give the impression of typing input in one line
            key = getkey()
            if key == keys.ESC:
                user_input = ""
                is_running = False
                break
            elif key == keys.BACKSPACE:
                user_input = user_input[:-1]
                print(f"  -> {user_input} ", end='\r') # Put this here to fix lingering UI issue
            elif key == keys.TAB:
                print(f"     {len(user_input)*" "}", end='\r') # Put this here to fix lingering UI issue
                formatted_user_input = user_input.lower()
                suggestions = get_suggestions(formatted_user_input, pkmn_obj.names_trie)
                if suggestions:
                    print(f"Suggestion(s): {", ".join(suggestions)}")
                else:
                    print(f"Suggestions: None")
            elif key.isalnum() or key in SPECIAL_CHARS_IN_NAMES:
                user_input += key
        print("\n") # Put this here to fix missing newline UI issue

        # 3rd part of loop: Take in user input and check if the requested Pokemon's data exists
        # If so, continue to 4th part, else restart to 1st part again
        if not user_pokemon_data and user_input:
            # If user only typed in integers, check for pokemon by pokedex id
            if user_input.isdigit():
                int_val = int(user_input)
                if int_val < 1 or int_val > pkmn_obj.max_pokemon:
                    print(f"\nPokeDex: There is no such pokemon with pokedex number '{user_input}'. Try a number between 1 and {pkmn_obj.max_pokemon}.")
                else:
                    user_pokemon_data = pkmn_obj.csv_data[int_val-1]
            # Else, check for pokemon by name
            else:
                formatted_input = user_input.lower()
                for line in pkmn_obj.csv_data:
                    if formatted_input == line[PKMN_DATA.CSV_NAME_KEY].lower():
                        user_pokemon_data = line
                        break
                if not user_pokemon_data:
                    print(f"\nPokeDex: There is no such pokemon named '{user_input}'.")

        # 4th part of loop: Retrieve data from valid Pokemon selection
        if user_pokemon_data and not retriever:
            # Print this statement to buy some time while loading the chosen pokemon's web data
            print(f"\nContacting {user_pokemon_data[PKMN_DATA.CSV_NAME_KEY]}...")

            web_paths.extend(user_pokemon_data[PKMN_DATA.CSV_LINKS_KEY])
            loader = WebBaseLoader(
                web_paths=(web_paths),
                requests_per_second=2
            )
            all_data = loader.load()

            # Remove files in extra files folder before continuing
            for file in os.listdir(PKMN_DATA.EXTRA_FILES_DIR):
                file_path = os.path.join(PKMN_DATA.EXTRA_FILES_DIR, file)
                os.remove(file_path)

            # Since these links have access restrictions, grab content via HTTP requests instead of WebBaseLoader and remove HTML chars
            for other_link in user_pokemon_data[PKMN_DATA.CSV_OTHER_LINKS_KEY]:
                response = requests.get(other_link)
                soup = BeautifulSoup(response.text, "html.parser")
                text_data = soup.get_text()

                # Write data as a text file and append its content into the document object
                other_file_path = os.path.join(PKMN_DATA.EXTRA_FILES_DIR, user_pokemon_data[PKMN_DATA.CSV_NAME_KEY] + '.txt')
                with open(other_file_path, "w") as f:
                    f.write(text_data)
                curr_loader = TextLoader(other_file_path)
                all_data += curr_loader.load()

            # Split data into chunks for easier lookup
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200, add_start_index=True
            )
            split_all_data = text_splitter.split_documents(all_data)

            # Store in a vector database for easier querying
            vectorstore = Chroma.from_documents(documents=split_all_data, embedding=OpenAIEmbeddings())

            # Set the retriever and filtering parameters
            retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 20})

            # TODO: Figure how out how to get OpenAI to consistently start with this intro so I don't need to hardcode the rest of this section below!
            # Always start with this Intro:
            intro_text = f"Hello, my name is {user_pokemon_data[PKMN_DATA.CSV_NAME_KEY]}. I am a {user_pokemon_data[PKMN_DATA.CSV_TYPE_KEY]} type pokemon from gen {str(user_pokemon_data[PKMN_DATA.CSV_GEN_KEY])}. What would you like to ask me?"
            
            # If pokemon speaks telepathically, showcase it
            if user_pokemon_data[PKMN_DATA.CSV_SPEECH_KEY] == "telepathy":
                intro_text = "*Speaking telepathically* " + intro_text
            # Else if pokemon speaks through pokemon noises, showcase it
            elif user_pokemon_data[PKMN_DATA.CSV_SPEECH_KEY] == "noise":
                intro_text = "*" + user_pokemon_data[PKMN_DATA.CSV_NAME_KEY] + " noises* (" + intro_text + ")"

            print(f"\n{user_pokemon_data[PKMN_DATA.CSV_NAME_KEY]} (#{str(user_pokemon_data[PKMN_DATA.CSV_ID_KEY])}): {intro_text}")

        # 5th part of loop: Converse with user until current conversation ends
        while retriever and ENDING_PHRASE not in result:
            user_input = input("  -> ")

            # Convert retriever obj's content into a list of strings to better
            # append to prompt as additional context when generating a response
            relevant_context = ""
            relevant_docs = retriever.invoke(user_input)
            for idx, doc in enumerate(relevant_docs):
                relevant_context += f"{idx+1}. {doc.page_content}\n"

            # Insert prompt, input and formmatting into chat model and print the response
            response_count += 1
            chain = chat_prompt | chat_model | CustomParser()

            # Be prepared to handle OpenAI not working... for a variety of reasons
            try:
                result = chain.invoke(
                    {"pokemon_name":user_pokemon_data[PKMN_DATA.CSV_NAME_KEY],
                    "type":user_pokemon_data[PKMN_DATA.CSV_NAME_KEY],
                    "gen":user_pokemon_data[PKMN_DATA.CSV_GEN_KEY],
                    "text":user_input,
                    "end_phrase":ENDING_PHRASE,
                    "count":response_count,
                    "context":relevant_context,
                    "chat_history":"\n".join(chat_history)}
                )
            except Exception as e:
                error_msg = "I'm not sure what the issue is but maybe wait a bit"
                if e.status_code == 401:
                    error_msg = "I believe you're using an invalid API Key or not part of the OpenAI organization. Please check your API key"
                elif e.status_code == 403:
                    error_msg = "I believe you're trying to use OpenAI in an unsupported location. Please change locations"
                elif e.status_code == 429:
                    error_msg = "I believe you're either sending OpenAI requests to quickly or are out of funds. Please check your API usage"
                elif e.status_code == 500 or e.status_code == 503:
                    error_msg = "I believe this was a server error. Please wait a bit"
                print(f"\nPokeDex: Sorry, I lost contact with {user_pokemon_data[PKMN_DATA.CSV_NAME_KEY]}. {error_msg} and then try running me again.")
                break
            print(result)

            # keep track of the 20 most recent inputs & responses
            chat_history.append("User: " + user_input)
            chat_history.append("You: " + result)
            chat_history = chat_history[-20:]

    print("\nPokeDex: Goodbye!")
