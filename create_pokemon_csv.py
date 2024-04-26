import csv
import os

import bs4
from langchain_community.document_loaders import WebBaseLoader


DIR = os.path.dirname(__file__)
PDF_DIR = os.path.join(DIR, "PDFs")
CSV_FILE_PATH = os.path.join(DIR, "Pokemon.csv")
VERSION_FILE_PATH = os.path.join(DIR, "Version.txt")
BULBAPEDIA_FILE_PATH = os.path.join(DIR, "Bulbapedia_Source_Code.txt")

CSV_ID_KEY = "ID"
CSV_NAME_KEY = "Name"
CSV_TYPE_KEY = "Type"
CSV_SPEECH_KEY = "Speech"
CSV_GEN_KEY = "Generation"
CSV_LINKS_KEY = "Links"
CSV_PDF_LINKS_KEY = "PDF_Links"
CSV_FIELDS =  [CSV_ID_KEY,CSV_NAME_KEY,CSV_TYPE_KEY,CSV_SPEECH_KEY,CSV_GEN_KEY,CSV_LINKS_KEY,CSV_PDF_LINKS_KEY]

POKEMON_TYPES = ["normal", "fire", "water", "electric", "grass", "ice", "fighting", "poison",
"ground", "flying", "psychic", "bug", "rock", "ghost", "dragon", "dark", "steel", "fairy"]

PKMN_DB_WEB_PATH = "https://pokemondb.net/pokedex/national"
PKMN_TALK_WEB_PATH = "https://pallettown.fandom.com/wiki/Talking_Pok%C3%A9mon"

SPLIT_PKMN_LIST_STR = "<div class=\"infocard\"><span class=\"infocard-lg-img\">"
SPLIT_PKMN_DATA_STR = "<a class=\"ent-name\" href=\"/pokedex/"
SPLIT_PKMN_TYPE_STR = "<a class=\"itype "
SPLIT_BULBA_LIST_STR = "<td><a href=\"{\\field{\\*\\fldinst{HYPERLINK \""

# Increment this everytime there are csv changes or updates from data websites
CURRENT_VERSION = 2


class PokemonData():
    """
    Object for holding all downloaded Pokemon data
    """
    csv_data: list[dict]
    max_pokemon: int
    _version: int

    def __init__(self):
        self.load()
        if not self.is_current_version():
            self.retrieve_and_write()

    @property
    def version(self) -> int:
        return self._version
    
    @version.setter
    def version(self, value: int):
        self._version = value

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
                    self.csv_data.append(data)
                    pokemon_count += 1
                self.max_pokemon = pokemon_count

            with open(VERSION_FILE_PATH, mode ='r') as f:
                self.version = int(f.read())
        except:
            if os.path.exists(CSV_FILE_PATH):
                os.remove(CSV_FILE_PATH)
            if os.path.exists(VERSION_FILE_PATH):
                os.remove(VERSION_FILE_PATH)
            self.version = -1
            return

    def retrieve_and_write(self):
        """
        Grabs data from oneline websites and encapsulates important
        data into a csv data table
        """
        self._get_data()

        if os.path.exists(CSV_FILE_PATH):
            os.remove(CSV_FILE_PATH)

        if os.path.exists(VERSION_FILE_PATH):
            os.remove(VERSION_FILE_PATH)

        if not os.path.isdir(PDF_DIR):
            os.mkdir(PDF_DIR)

        with open(CSV_FILE_PATH, 'w') as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(self.csv_data)

        with open(VERSION_FILE_PATH, 'w') as f:
            f.write(str(CURRENT_VERSION))

        # TODO: Come up with a better fix
        # This is gross but data must be loaded up the same way everytime to resolve consistency issues
        self.load()

    def is_current_version(self) -> bool:
        return self.version == CURRENT_VERSION

    def _get_data(self):
        pokemon_data = []

        # Start by grabbing special pokemon speech data from a specific website since it will be needed later
        bs4_filter = bs4.SoupStrainer("ul")
        loader = WebBaseLoader(
            web_path=(PKMN_TALK_WEB_PATH),
            bs_kwargs={"parse_only": bs4_filter}
        )
        # Due to how this data is currently retrieved, oddly-specific storing and spliting are needed to ensure proper labeling
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
        # retrieve pokemon links from Bulbapedia's list of pokemon (source code)
        # and store them to later convert them into PDFs to extract their content
        pdf_links = []
        f = open(BULBAPEDIA_FILE_PATH, "r")
        parsed_pokemon_links = f.read().split(SPLIT_BULBA_LIST_STR)[1:]
        # All parsed data here has the exact same format:
        # [pokemon_link]"...
        for pokemon_link in parsed_pokemon_links:
            end_idx = pokemon_link.find("\"")
            pdf_link = pokemon_link[:end_idx]
            # Don't include any duplicate links
            if pdf_link not in pdf_links:
                pdf_links.append(pdf_link)

        # Next, get list of all current pokemon from the online pokemondb database
        # website and filter out website content to just pokemon names and types
        class_types = ["itype " + x for x in POKEMON_TYPES]
        class_types.append("ent-name")
        class_types.append("infocard-list infocard-list-pkmn-lg")
        bs4_filter = bs4.SoupStrainer(class_=(class_types))
        loader = WebBaseLoader(
            web_path=(PKMN_DB_WEB_PATH),
            bs_kwargs={"parse_only": bs4_filter}
        )
        pkmndb_web_data = loader.scrape()
    
        # Now sort through data and store pokemon ids, names, types, speech patterns,
        # generations & urls from retrieving additional information
        id = 0
        gen = 0
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
            
                # All parsed data here has the exact same format:
                # [child_url]">'[pokemon_name]...<a class="itype [pokemon_type1]"...
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
                
                # Next, use talking pokemon data to determine speech attribute
                # Due to how this data is currently parsed, oddly-specific conditions are needed to ensure proper labeling
                pokemon_speech = "noise"
                pokemon_name_lowercase = pokemon_name.lower()

                # Specifically skip Mew since it otherwise gets mislabeled with Mewtwo's speech value
                if pokemon_name_lowercase != "mew":
                    # If current pokemon is mentioned in talk data, find it's correct speech value
                    for talk_data in talk_data_list:
                        if pokemon_name_lowercase in talk_data:
                            # Edge Case: some talking pokemon have "without telepathy" in their description
                            if "without telepathy" in talk_data:
                                pokemon_speech = "talk"

                            # Any mentions of "telepath" means the pokemon speaks through telepathy
                            # Edge Case: Sometimes telepathy is implied with "talk through meowth"
                            elif ("telepath" in talk_data or "talk through meowth" in talk_data) and pokemon_speech != "talk":
                                pokemon_speech = "telepathy"

                            # If there's no mention of "telepathy", then the pokemon can actually speak
                            else:
                                pokemon_speech = "talk"

                # Lastly, grab the current pokemon's pdf link. Since the pdf link list is already
                # in pokedex order, just pop the current top link from the list
                pokemon_pdf_link = []
                pokemon_pdf_link.append(pdf_links.pop(0))

                # Now package it into a nice dict and append to pokemon list
                dict_data = {
                    CSV_ID_KEY:id,
                    CSV_NAME_KEY:pokemon_name,
                    CSV_TYPE_KEY:pokemon_type,
                    CSV_SPEECH_KEY:pokemon_speech,
                    CSV_GEN_KEY:gen,
                    CSV_LINKS_KEY:pokemon_name_entry(pokemon_url,id),
                    CSV_PDF_LINKS_KEY:pokemon_pdf_link
                }
                pokemon_data.append(dict_data)

        self.csv_data = pokemon_data
        self.max_pokemon = id
        self.version = CURRENT_VERSION


def pokemon_name_entry(name: str, id: int) -> list:
    websites = []
    str_num = str(id)
    
    # A formatted string should be 4 digits
    str_num_formmatted = str_num
    for i in range(4 - len(str_num)):
        str_num_formmatted = "0" + str_num_formmatted

    # Append these different pokemon database sites
    websites.append("https://pokemondb.net/pokedex/" + name)
    websites.append("https://pokedex.org/#/pokemon/" + str_num)
    websites.append("https://sg.portal-pokemon.com/play/pokedex/" + str_num_formmatted)

    return websites


# Keeping this just in case it's needed later
def pokemon_type_entry(type: str) -> str:
    return "https://pokemondb.net/type/" + type
