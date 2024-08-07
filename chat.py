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
ENDING_PHRASE = " Thank you for talking to me!"
MAX_SUGGESTIONS = 10
MAX_RESPONSES = 10


class CustomParser(BaseOutputParser):
    """
    This parser object allows users to format a chat model's response.
    """
    def parse(self, text: str) -> str:
        # Filter out newlines and multiple whitespaces for a cleaner text display
        formatted_text = text.replace("\n"," ")
        while "  " in formatted_text:
            formatted_text.replace("  ", " ")
        return formatted_text


# NOTE: This function was separated from the parser class above to address a formatting bug in the model's responses
# Print out model response in this format -> [pokemon_name]([pokedex_id]): [speech]
# Include the pokemon's speech pattern (ex. telepath, nosies, etc.) as well
def print_response(pkmn_data: dict, text: str):
    response = f"\n{pkmn_data[PKMN_DATA.CSV_NAME_KEY]} (#{str(pkmn_data[PKMN_DATA.CSV_ID_KEY])}): "
    if pkmn_data[PKMN_DATA.CSV_SPEECH_KEY] == "telepathy":
        response += ("*Speaking telepathically* " + text)
    elif pkmn_data[PKMN_DATA.CSV_SPEECH_KEY] == "noise":
        response += ("*" + pkmn_data[PKMN_DATA.CSV_NAME_KEY] + " noises* (" + text + ")")
    else:
        response += text
    print(response)


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

    # Randomly sort suggestions and cap amount if necessary
    num_suggestions = min(len(suggestions), MAX_SUGGESTIONS)
    return sample(suggestions, num_suggestions)


if __name__ == "__main__":
    # Print this statement to buy some time while setting up data & model
    print("Contacting PokeDex...")

    # Load up hidden API key located in a different file and set up chatbot model
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    
    # Retrieve pokemon data from websites
    pkmn_obj = PKMN_DATA.PokemonData()

    # Assign prompt & inputs for Pokemon conversations
    chat_model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=api_key)

    system_template = """
        You are a Pokemon named {pokemon_name}, a {type} type from generation {gen}. You are having a conversation
        with a user who wants to learn more about you and the Pokemon world you live in. Utilize the "Conversation
        Rules", "Important Information" and "Chat History" sections listed below when responding to the user.

        Conversation Rules:
        1.  Keep your responses to 3 sentences max unless the user explicitly gives you a response length to follow.
        2.  Do not lie or make up answers. Use the information found in "Important Information" as additional context
            (if possible) when deciding on a response.
        3.  Use the conversation history in "Chat History" as additional context when evaluating the user's input.
        4.  If the user asks you something and you don't know the answer, let the user know you don't know and ask
            another question about yourself.
        
        Important Information:
        {context}
        
        Chat History:
        {chat_history}
    """
    human_template = "{text}"

    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
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
        chat_result = ""
        key = ""
        response_count = 0

        # 2nd part of loop: Ensure either a pokemon is selected to converse with or user is done
        print("\nPokeDex: Please type in the pokemon's name or pokedex number to get started, press the TAB key "
              "for a suggestion based on current input or press the ESCAPE key to exit.")
        # Use getKey() instead of input() to offer suggestions as the user types a name and filter out invalid chars
        # If there's only one suggestion, autocomplete the user's input with it
        # ESC -> exit program, BACKSPACE -> delete newest char, TAB -> give suggestion(s), ENTER -> accept user input
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
                if len(suggestions) == 1:
                    user_input += suggestions[0][len(user_input):]
                elif suggestions:
                    print(f"Suggestions: {", ".join(suggestions)}")
                else:
                    print(f"Suggestions: NONE")
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
                    print(f"\nPokeDex: There is no such pokemon with pokedex number '{user_input}'."
                          f" Try a number between 1 and {pkmn_obj.max_pokemon}.")
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

            # Since these links have access restrictions, grab content via HTTP requests instead of WebBaseLoader
            # Also remove all HTML characters after a successful retrieval
            for other_link in user_pokemon_data[PKMN_DATA.CSV_OTHER_LINKS_KEY]:
                response = requests.get(other_link)
                soup = BeautifulSoup(response.text, "html.parser")
                text_data = soup.get_text()

                # Write data as a text file and append its content into the document object
                other_file_path = os.path.join(PKMN_DATA.EXTRA_FILES_DIR,
                    user_pokemon_data[PKMN_DATA.CSV_NAME_KEY] + '.txt')
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

        # 5th part of loop: Converse with user until current conversation ends
        while retriever and response_count < MAX_RESPONSES:
            response_count += 1
            
            # Skip user input on the very first response since we need to produce a prompt first
            if response_count != 1:
                user_input = input("  -> ")
            else:
                user_input = "Tell me about yourself (in the first person point of view) and include " + \
                    "a suggestion to ask you a question about yourself or the Pokemon world you live in."

            # Convert retriever obj's content into a list of strings to better
            # append to prompt as additional context when generating a response
            relevant_context = ""
            relevant_docs = retriever.invoke(user_input)
            for idx, doc in enumerate(relevant_docs):
                relevant_context += f"{idx+1}. {doc.page_content}\n"

            # Insert prompt, input and formmatting into chat & tagging models
            chat_chain = chat_prompt | chat_model | CustomParser()

            # Be prepared to handle OpenAI not working... for a variety of reasons
            try:
                chat_result = chat_chain.invoke(
                    {"pokemon_name":user_pokemon_data[PKMN_DATA.CSV_NAME_KEY],
                    "type":user_pokemon_data[PKMN_DATA.CSV_TYPE_KEY],
                    "gen":user_pokemon_data[PKMN_DATA.CSV_GEN_KEY],
                    "text":user_input,
                    "context":relevant_context,
                    "chat_history":"\n".join(chat_history)}
                )

                # Check if current conversation is done
                if response_count >= MAX_RESPONSES:
                    chat_result += ENDING_PHRASE

            except Exception as e:
                # Establish "technical issue" with the PokeDex
                context_msg = "Sorry, I lost contact with"
                if response_count == 1:
                    context_msg = "Sorry, I was unable to contact"
                
                # Determine cause of OpenAI issue
                error_msg = "I'm not sure what the issue is but maybe wait a bit"
                if e.status_code == 401:
                    error_msg = "You're using an invalid or non-OpenAI affiliated API Key. Please check your API key"
                elif e.status_code == 403:
                    error_msg = "You're trying to use OpenAI in an unsupported location. Please change locations"
                elif e.status_code == 429:
                    error_msg = "You're either sending OpenAI requests to quickly or are out of funds. Please fix this"
                elif e.status_code == 500 or e.status_code == 503:
                    error_msg = "I believe this was a server error. Please wait a bit"

                # Provide an explanation to hint the user on how to resolve the issue and exit
                print(f"\nPokeDex: {context_msg} {user_pokemon_data[PKMN_DATA.CSV_NAME_KEY]}."
                      f" {error_msg} and then try running me again.")
                is_running = False
                break

            else:
                # Print successfully generated response in the desired format
                print_response(user_pokemon_data, chat_result)

                # keep track of the current inputs & responses
                chat_history.append("User Input: " + user_input)
                chat_history.append("Your Output: " + chat_result)

    print("\nPokeDex: Goodbye!")
