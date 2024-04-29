import os

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import requests

import create_pokemon_csv as PKMN_CSV


ENDING_PHRASE = "Thank you for talking to me!"
user_pokemon_data = {} # Need to make this global so parser class can access it


class CustomParser(BaseOutputParser):
    """
    This parser object allows users to format a chat model's response.
    """
    def parse(self, text: str) -> str:
        # Assume pokemon speaks normally unless speech value says otherwise
        formatted_text = text
        
        # If pokemon speaks telepathically, showcase it
        if user_pokemon_data[PKMN_CSV.CSV_SPEECH_KEY] == "telepathy":
            formatted_text = "*Speaking telepathically* " + formatted_text

        # Else if pokemon speaks through pokemon noises, showcase it
        elif user_pokemon_data[PKMN_CSV.CSV_SPEECH_KEY] == "noise":
            formatted_text = "*" + user_pokemon_data[PKMN_CSV.CSV_NAME_KEY] + " noises* (" + formatted_text + ")"

        # Format should always be: [pokemon_name]([pokedex_id]): speech
        return "\n" + user_pokemon_data[PKMN_CSV.CSV_NAME_KEY] + " (#" + user_pokemon_data[PKMN_CSV.CSV_ID_KEY] + "): " + formatted_text


if __name__ == "__main__":
    # Print this statement to buy some time while setting up data & model
    print("Contacting PokeDex...")

    # Load up hidden API key located in a different
    # file and set up chatbot model
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    chat_model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=api_key)
    
    # Retrieve pokemon data from websites
    pkmn_obj = PKMN_CSV.PokemonData()

    # Assign prompt & input
    template = """
        You are {pokemon_name}, a {type} type pokemon from generation {gen}. You are talking to a user who wants to ask you questions about yourself or give you requests. Follow the conversation rules listed below when responding to the user.

        Conversation Rules:
        1. Keep your responses to 3 sentences max unless the user explicitly gives you a response length to follow.
        2. If the user asks something inappropriate, reply with "I don't feel comfortable with this conversation. Would you like to talk about something else instead?".
        3. If the user's input is incomprehensible, reply with "I don't understand. Would you like to ask me something else?".
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

    chat_history = []
    retriever = ""
    pkmn_typing = ""
    response_count = 0
    is_running = True
    while True:
        # Ensure a pokemon is selected to converse with
        while not user_pokemon_data:
            web_paths = [PKMN_CSV.PKMN_DB_WEB_PATH]
            user_input = input("\nPokeDex: Please type in the pokemon's name or pokedex number to get started or type 'exit' if you're done.\n  -> ")
            if not user_input:
                continue
            elif "exit" in user_input:
                is_running = False
                break
            # If user (only) typed in a number, check for pokemon by pokedex id
            elif user_input.isdigit():
                int_val = int(user_input)
                if int_val == 0 or int_val > pkmn_obj.max_pokemon:
                    print(f"\nPokeDex: There is no such pokemon with that pokedex number. Try a number between 1 and {pkmn_obj.max_pokemon}.")
                    continue
                for line in pkmn_obj.csv_data:
                    if int_val == int(line[PKMN_CSV.CSV_ID_KEY]):
                        user_pokemon_data = line
                        break
            # Else, check for pokemon by name
            else:
                formatted_input = user_input.lower()
                for line in pkmn_obj.csv_data:
                    if formatted_input == line[PKMN_CSV.CSV_NAME_KEY].lower():
                        user_pokemon_data = line
                        break
                if not user_pokemon_data:
                    print("\nPokeDex: There is no such pokemon with that name.")

            # Once user has selected a valid pokemon, retrieve their additional data
            # and start up conversation with the chosen pokemon
            if user_pokemon_data:
                # Print this statement to buy some time while loading the chosen pokemon's web data
                print(f"\nContacting {user_pokemon_data[PKMN_CSV.CSV_NAME_KEY]}...")
                
                # TODO: Look into and fix csv/extra files loading issue?
                # The links list comes back as a string when uploading csv data unfortunately...fixing it here
                formatted_links_list = user_pokemon_data[PKMN_CSV.CSV_LINKS_KEY][1:-1].replace("'","").split(", ")
                web_paths.extend(formatted_links_list)

                loader = WebBaseLoader(
                    web_paths=(web_paths),
                    requests_per_second=2
                )
                all_data = loader.load()

                # Remove files in extra files folder before continuing
                for file in os.listdir(PKMN_CSV.EXTRA_FILES_DIR):
                    file_path = os.path.join(PKMN_CSV.EXTRA_FILES_DIR, file)
                    os.remove(file_path)

                # Since these links have access restrictions, grab content via HTTP requests instead of WebBaseLoader
                # The other links list also comes back as a string when uploading csv data unfortunately...fixing it here
                formatted_other_links_list = user_pokemon_data[PKMN_CSV.CSV_OTHER_LINKS_KEY][1:-1].replace("'","").split(", ")
                for other_link in formatted_other_links_list:
                    # Get content from link and remove HTML characters
                    response = requests.get(other_link)
                    soup = BeautifulSoup(response.text, "html.parser")
                    text_data = soup.get_text()

                    # Write data as a text file and append its content into the document object
                    other_file_path = os.path.join(PKMN_CSV.EXTRA_FILES_DIR, user_pokemon_data[PKMN_CSV.CSV_NAME_KEY] + '.txt')
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

                # TODO: Figure how out how to get OpenAI to consistently start with this intro so I don't need to hardcode it here!
                # Always start with this Intro:
                intro_text = f"Hello, my name is {user_pokemon_data[PKMN_CSV.CSV_NAME_KEY]}. I am a {user_pokemon_data[PKMN_CSV.CSV_TYPE_KEY]} type pokemon from gen {user_pokemon_data[PKMN_CSV.CSV_GEN_KEY]}. What would you like to ask me?"
                
                # If pokemon speaks telepathically, showcase it
                if user_pokemon_data[PKMN_CSV.CSV_SPEECH_KEY] == "telepathy":
                    intro_text = "*Speaking telepathically* " + intro_text
                # Else if pokemon speaks through pokemon noises, showcase it
                elif user_pokemon_data[PKMN_CSV.CSV_SPEECH_KEY] == "noise":
                    intro_text = "*" + user_pokemon_data[PKMN_CSV.CSV_NAME_KEY] + " noises* (" + intro_text + ")"

                print(f"\n{user_pokemon_data[PKMN_CSV.CSV_NAME_KEY]} (#{user_pokemon_data[PKMN_CSV.CSV_ID_KEY]}): {intro_text}")

        if not is_running:
            break

        # Converse with user
        user_input = ""
        while not user_input:
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
        result = chain.invoke(
            {"pokemon_name":user_pokemon_data[PKMN_CSV.CSV_NAME_KEY],
            "type":pkmn_typing,
            "gen":user_pokemon_data[PKMN_CSV.CSV_GEN_KEY],
            "text":user_input,
            "end_phrase":ENDING_PHRASE,
            "count":response_count,
            "context":relevant_context,
            "chat_history":"\n".join(chat_history)}
        )
        print(result)

        # keep track of the 20 most recent inputs & responses
        chat_history.append("User: " + user_input)
        chat_history.append("You: " + result)
        chat_history = chat_history[-20:]

        # End current conversation and reset variables
        if ENDING_PHRASE in result:
            user_pokemon_data = {}
            chat_history = []
            retriever = ""
            pkmn_typing = ""
            response_count = 0

    print("\nPokeDex: Understood, goodbye!")
