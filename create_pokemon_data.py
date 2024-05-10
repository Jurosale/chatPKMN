import csv
import json
import os

from bs4 import BeautifulSoup, SoupStrainer
from langchain_community.document_loaders import WebBaseLoader
import requests


DIR = os.path.dirname(__file__)
GENERATED_DIR = os.path.join(DIR, "GENERATED")
EXTRA_FILES_DIR = os.path.join(GENERATED_DIR, "Extra_Files")
CSV_FILE_PATH = os.path.join(GENERATED_DIR, "Pokemon.csv")
VERSION_FILE_PATH = os.path.join(GENERATED_DIR, "Version.txt")
NAMES_TRIE_FILE_PATH = os.path.join(GENERATED_DIR, "Names_Trie.json")

CSV_ID_KEY = "ID"
CSV_NAME_KEY = "Name"
CSV_TYPE_KEY = "Type"
CSV_SPEECH_KEY = "Speech"
CSV_GEN_KEY = "Generation"
CSV_LINKS_KEY = "Links"
CSV_OTHER_LINKS_KEY = "Other_Links"
CSV_FIELDS =  [CSV_ID_KEY,CSV_NAME_KEY,CSV_TYPE_KEY,CSV_SPEECH_KEY,CSV_GEN_KEY,CSV_LINKS_KEY,CSV_OTHER_LINKS_KEY]

POKEMON_TYPES = ["normal", "fire", "water", "electric", "grass", "ice", "fighting", "poison",
"ground", "flying", "psychic", "bug", "rock", "ghost", "dragon", "dark", "steel", "fairy"]

PKMN_DB_WEB_PATH = "https://pokemondb.net/pokedex/national"
PKMN_TALK_WEB_PATH = "https://pallettown.fandom.com/wiki/Talking_Pok%C3%A9mon"
PKMN_BULBA_WEB_PATH = "https://bulbapedia.bulbagarden.net/wiki/List_of_Pok%C3%A9mon_by_National_Pok%C3%A9dex_number"

SPLIT_PKMN_LIST_STR = "<div class=\"infocard\"><span class=\"infocard-lg-img\">"
SPLIT_PKMN_DATA_STR = "<a class=\"ent-name\" href=\"/pokedex/"
SPLIT_PKMN_TYPE_STR = "<a class=\"itype "
SPLIT_BULBA_LIST_STR = "<td><a href=\""

END_NAME_STR = "\n"

# Increment this everytime there are csv changes or updates from data websites
CURRENT_VERSION = 5


class PokemonData():
    """
    Object for holding all downloaded Pokemon data
    """
    _csv_data: list[dict]
    _names_trie: dict
    _max_pokemon: int
    _version: int

    def __init__(self):
        self.load()
        if not self.is_current_version():
            self.retrieve_and_write()

    @property
    def csv_data(self) -> list[dict]:
        return self._csv_data

    @csv_data.setter
    def csv_data(self, data: list[dict]):
        if isinstance(data, list):
            self._csv_data = data
        else:
            print("data is not a list. Not setting as csv data.")

    @property
    def names_trie(self) -> dict:
        return self._names_trie

    @names_trie.setter
    def names_trie(self, trie: dict):
        if isinstance(trie, dict):
            self._names_trie = trie
        else:
            print("data is not a dict. Not setting as names trie.")

    @property
    def max_pokemon(self) -> int:
        return self._max_pokemon

    @max_pokemon.setter
    def max_pokemon(self, value: int):
        if isinstance(value, int) and value >= 0:
            self._max_pokemon = value
        else:
            print("value is not a non-negative integer. Not setting as max pokemon.")

    @property
    def version(self) -> int:
        return self._version

    @version.setter
    def version(self, value: int):
        if isinstance(value, int):
            self._version = value
        else:
            print("value is not an integer. Not setting as version.")

    def load(self):
        """
        Loads data from generated csv path. If it's missing or fails
        for any reason, it will generate a new one
        """
        try:
            with open(CSV_FILE_PATH, mode ='r') as f:
                pokemon_count = 0
                self.csv_data = []
                for data in csv.DictReader(f):
                    # Unfortunately csv.writer converted all values into strings, fixing that here
                    data[CSV_ID_KEY] = int(data[CSV_ID_KEY]) # int
                    data[CSV_GEN_KEY] = int(data[CSV_GEN_KEY]) # int
                    data[CSV_LINKS_KEY] = data[CSV_LINKS_KEY][1:-1].replace("'","").split(", ") # list
                    data[CSV_OTHER_LINKS_KEY] = data[CSV_OTHER_LINKS_KEY][1:-1].replace("'","").split(", ") # list

                    self.csv_data.append(data)
                    pokemon_count += 1
                self.max_pokemon = pokemon_count

            with open(VERSION_FILE_PATH, mode ='r') as f:
                self.version = int(f.read())

            with open(NAMES_TRIE_FILE_PATH, mode ='r') as f:
                self.names_trie = json.loads(f.read())
        except:
            self.version = -1
            return

    def retrieve_and_write(self):
        """
        Grabs data from oneline websites and encapsulates important
        data into a csv data table
        """
        self._get_data()

        # Created GENERATED folder and extra files subdirectory if they don't already exist
        if not os.path.isdir(GENERATED_DIR):
            os.mkdir(GENERATED_DIR)

        if not os.path.isdir(EXTRA_FILES_DIR):
            os.mkdir(EXTRA_FILES_DIR)

        # Remove outdated generated files
        if os.path.exists(CSV_FILE_PATH):
            os.remove(CSV_FILE_PATH)

        if os.path.exists(VERSION_FILE_PATH):
            os.remove(VERSION_FILE_PATH)

        if os.path.exists(NAMES_TRIE_FILE_PATH):
            os.remove(NAMES_TRIE_FILE_PATH)

        # Create new generated files with up-to-date info
        with open(CSV_FILE_PATH, 'w') as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(self.csv_data)

        with open(VERSION_FILE_PATH, 'w') as f:
            f.write(str(self.version))

        with open(NAMES_TRIE_FILE_PATH, 'w') as f:
            json_str = json.dumps(self.names_trie)
            f.write(json_str)

    def is_current_version(self) -> bool:
        return self.version == CURRENT_VERSION

    def _get_data(self):
        pokemon_data = []

        # Start by grabbing special pokemon speech data from a specific website since it will be needed later
        bs4_filter = SoupStrainer("ul")
        loader = WebBaseLoader(
            web_path=(PKMN_TALK_WEB_PATH),
            bs_kwargs={"parse_only": bs4_filter}
        )
        # Due to how this data is currently retrieved, oddly-specific storing and spliting techniques are
        # needed to ensure proper labeling of speech attributes
        pkmn_talk_web_data = str(loader.scrape().contents[4])
        talk_data_list = []
        # Need to retrieve only actual text from site
        for entry in pkmn_talk_web_data.split("<li>")[1:]:
            # Start by removing titled movies/episodes to avoid mislabeling pokemon speech
            start_char_idx = entry.find("<i>")
            while start_char_idx != -1:
                end_char_idx = entry.find("</i>")
                entry = entry[:start_char_idx] + entry[end_char_idx+4:]
                start_char_idx = entry.find("<i>")

            # Now remove web chars so only actual text remains
            start_char_idx = entry.find("<")
            while start_char_idx != -1:
                end_char_idx = entry.find(">")
                entry = entry[:start_char_idx] + entry[end_char_idx+1:]
                start_char_idx = entry.find("<")
            talk_data_list.append(entry.lower())

        # Now as a workaround to needing permission to access Bulbapedia's website:
        # retrieve pokemon links from Bulbapedia's list of pokemon using an HTTP request
        # and store them to later extract their content in a different manner
        other_links = []
        response = requests.get(PKMN_BULBA_WEB_PATH)
        soup = BeautifulSoup(response.text, "html.parser")
        bulba_data = ""
        # All bulbapedia's pokemon links exist within "td" tags
        for data in soup.find_all("td"):
            bulba_data += str(data)
        parsed_pokemon_links = bulba_data.split(SPLIT_BULBA_LIST_STR)[1:]
        # All parsed data here has the exact same format:
        # [pokemon_link]"...
        for pokemon_link in parsed_pokemon_links:
            end_idx = pokemon_link.find("\"")
            sublink = pokemon_link[:end_idx]
            other_link = pokemon_name_entry_bulbapedia(sublink)
            # Don't include any duplicate links
            if other_link not in other_links:
                other_links.append(other_link)

        # Next, get list of all current pokemon from the online pokemondb database
        # website and filter out website content to just pokemon names and types
        class_types = ["itype " + x for x in POKEMON_TYPES]
        class_types.append("ent-name")
        class_types.append("infocard-list infocard-list-pkmn-lg")
        bs4_filter = SoupStrainer(class_=(class_types))
        loader = WebBaseLoader(
            web_path=(PKMN_DB_WEB_PATH),
            bs_kwargs={"parse_only": bs4_filter}
        )
        pkmndb_web_data = loader.scrape()
    
        # Now sort through data and store pokemon ids, names, types, speech patterns,
        # generations & webpage urls for retrieving additional information
        id = 0
        gen = 0
        trie = {}

        # Due to how this website is specifically parsed, the pokemon content is
        # grouped by generation and thus each section also needs to be iterated through
        for data in pkmndb_web_data.contents:
            gen += 1
            str_data = str(data)
            pokemon_list_data= str_data.split(SPLIT_PKMN_LIST_STR)[1:]
            
            for curr_pokemon_data in pokemon_list_data:
                id += 1
                # Added "[0]" at the end so it returns a string instead of a list with a single, string entry
                parsed_data = curr_pokemon_data.split(SPLIT_PKMN_DATA_STR)[-1:][0]
            
                # All parsed data here have the exact same format:
                # [child_url]">[pokemon_name]...<a class="itype [pokemon_type1]"...
                # (optional): <a class="itype [pokemon_type2]"
                
                # Start by extracting child_url
                end_idx = parsed_data.find("\">")
                pokemon_url = parsed_data[:end_idx]
                parsed_data = parsed_data[end_idx+2:]
                
                # Now grab pokemon_name
                end_idx = parsed_data.find("<")
                pokemon_name = parsed_data[:end_idx]
                parsed_data = parsed_data[end_idx:]
                
                # Then grab types from remaining data
                parsed_type_data = parsed_data.split(SPLIT_PKMN_TYPE_STR)[1:]
                end_idx = parsed_type_data[0].find("\"")
                pokemon_type = parsed_type_data[0][:end_idx]
                # Some pokemon are dual type, format it as "type1/type2"
                if (len(parsed_type_data) == 2):
                    end_idx = parsed_type_data[1].find("\"")
                    pokemon_type += "/" + parsed_type_data[1][:end_idx]
                    
                # Make sure pokemon types are valid
                list_types = pokemon_type.split("/")
                for type in list_types:
                    if type.lower() not in POKEMON_TYPES:
                        raise Exception(f"{type} is not a valid type.")
                
                # Add name to trie data structure for suggestion purposes during runtime
                pokemon_name_lowercase = pokemon_name.lower()
                curr_dict = trie
                for char in pokemon_name_lowercase:
                    if char not in curr_dict:
                        curr_dict[char] = {}
                    curr_dict = curr_dict[char]
                # Add special char at the end to indicate a complete name
                curr_dict[END_NAME_STR] = {}

                # Next, use talking pokemon data and address the following edge cases to determine speech attribute:
                # Edge Case 1: any mentioned pokemon is considered a "talking" pokemon unless stated otherwise
                # Edge Case 2: skip Mew since it otherwise gets mislabeled with Mewtwo's speech value
                # Edge Case 3: some talking pokemon have "without telepathy" in their description
                # Edge Case 4: telepathy is stated or implied with "talk through meowth" in their description
                # Edge Case 5: some pokemon have multiple entries and "talking" has more priority than "telepathy"
                pokemon_speech = "noise"
                if pokemon_name_lowercase != "mew":
                    for talk_data in talk_data_list:
                        if pokemon_name_lowercase in talk_data:
                            if pokemon_speech != "talk" and "without telepathy" not in talk_data and \
                                ("telepath" in talk_data or "talk through meowth" in talk_data):
                                pokemon_speech = "telepathy"
                            else:
                                pokemon_speech = "talk"
                                break

                # Lastly, grab the current pokemon's other link (i.e. Bulbapedia entry). Since the other links
                # list is already in pokedex order, just pop the current top link from the list
                pokemon_other_links = []
                pokemon_other_links.append(other_links.pop(0))

                # Now package it into an organized dict and append to pokemon list
                dict_data = {
                    CSV_ID_KEY:id,
                    CSV_NAME_KEY:pokemon_name,
                    CSV_TYPE_KEY:pokemon_type,
                    CSV_SPEECH_KEY:pokemon_speech,
                    CSV_GEN_KEY:gen,
                    CSV_LINKS_KEY:pokemon_name_entry(pokemon_url,id),
                    CSV_OTHER_LINKS_KEY:pokemon_other_links
                }
                pokemon_data.append(dict_data)

        # Assign important class values
        self.csv_data = pokemon_data
        self.names_trie = trie
        self.max_pokemon = id
        self.version = CURRENT_VERSION


def pokemon_name_entry(name: str, id: int) -> list:
    websites = []
    str_num = str(id)
    
    # A formatted string should be 4 digits with leading zeros if necessary
    str_num_formmatted = str_num
    for i in range(4 - len(str_num)):
        str_num_formmatted = "0" + str_num_formmatted

    # Append these different pokemon database sites
    websites.append("https://pokemondb.net/pokedex/" + name)
    websites.append("https://pokedex.org/#/pokemon/" + str_num)
    websites.append("https://sg.portal-pokemon.com/play/pokedex/" + str_num_formmatted)

    return websites


def pokemon_name_entry_bulbapedia(name: str) -> str:
    return "https://bulbapedia.bulbagarden.net" + name


# Keeping this just in case it's needed later
def pokemon_type_entry(type: str) -> str:
    return "https://pokemondb.net/type/" + type
