from flask import Flask, render_template, request, jsonify
from math import gcd
from functools import reduce
from transformers import pipeline



app = Flask(__name__)

# Valency of all elements in the periodic table
valency = {
    # Group 1: Alkali Metals
    "H": 1,   # Hydrogen
    "Li": 1,  # Lithium
    "Na": 1,  # Sodium
    "K": 1,   # Potassium
    "Rb": 1,  # Rubidium
    "Cs": 1,  # Cesium
    "Fr": 1,  # Francium

    # Group 2: Alkaline Earth Metals
    "Be": 2,  # Beryllium
    "Mg": 2,  # Magnesium
    "Ca": 2,  # Calcium
    "Sr": 2,  # Strontium
    "Ba": 2,  # Barium
    "Ra": 2,  # Radium

    # Group 13: Boron Group
    "B": 3,   # Boron
    "Al": 3,  # Aluminum
    "Ga": 3,  # Gallium
    "In": 3,  # Indium
    "Tl": 3,  # Thallium

    # Group 14: Carbon Group
    "C": 4,   # Carbon
    "Si": 4,  # Silicon
    "Ge": 4,  # Germanium
    "Sn": 4,  # Tin
    "Pb": 4,  # Lead

    # Group 15: Nitrogen Group
    "N": 3,   # Nitrogen
    "P": 3,   # Phosphorus
    "As": 3,  # Arsenic
    "Sb": 3,  # Antimony
    "Bi": 3,  # Bismuth

    # Group 16: Oxygen Group
    "O": 2,   # Oxygen
    "S": 2,   # Sulfur
    "Se": 2,  # Selenium
    "Te": 2,  # Tellurium
    "Po": 2,  # Polonium

    # Group 17: Halogens
    "F": 1,   # Fluorine
    "Cl": 1,  # Chlorine
    "Br": 1,  # Bromine
    "I": 1,   # Iodine
    "At": 1,  # Astatine

    # Group 18: Noble Gases
    "He": 0,  # Helium (no valency)
    "Ne": 0,  # Neon (no valency)
    "Ar": 0,  # Argon (no valency)
    "Kr": 0,  # Krypton (no valency)
    "Xe": 0,  # Xenon (no valency)
    "Rn": 0,  # Radon (no valency)

    # Transition Metals
    "Sc": 3,  # Scandium
    "Ti": 4,  # Titanium
    "V": 5,   # Vanadium
    "Cr": 6,  # Chromium
    "Mn": 7,  # Manganese
    "Fe": 3,  # Iron (common valency: 2, 3)
    "Co": 2,  # Cobalt
    "Ni": 2,  # Nickel
    "Cu": 2,  # Copper
    "Zn": 2,  # Zinc
    "Y": 3,   # Yttrium
    "Zr": 4,  # Zirconium
    "Nb": 5,  # Niobium
    "Mo": 6,  # Molybdenum
    "Tc": 7,  # Technetium
    "Ru": 8,  # Ruthenium
    "Rh": 3,  # Rhodium
    "Pd": 2,  # Palladium
    "Ag": 1,  # Silver
    "Cd": 2,  # Cadmium
    "Hf": 4,  # Hafnium
    "Ta": 5,  # Tantalum
    "W": 6,   # Tungsten
    "Re": 7,  # Rhenium
    "Os": 8,  # Osmium
    "Ir": 3,  # Iridium
    "Pt": 2,  # Platinum
    "Au": 3,  # Gold
    "Hg": 2,  # Mercury
    "Rf": 4,  # Rutherfordium
    "Db": 5,  # Dubnium
    "Sg": 6,  # Seaborgium
    "Bh": 7,  # Bohrium
    "Hs": 8,  # Hassium
    "Mt": 9,  # Meitnerium
    "Ds": 10, # Darmstadtium
    "Rg": 11, # Roentgenium
    "Cn": 12, # Copernicium

    # Lanthanides
    "La": 3,  # Lanthanum
    "Ce": 4,  # Cerium
    "Pr": 3,  # Praseodymium
    "Nd": 3,  # Neodymium
    "Pm": 3,  # Promethium
    "Sm": 3,  # Samarium
    "Eu": 3,  # Europium
    "Gd": 3,  # Gadolinium
    "Tb": 3,  # Terbium
    "Dy": 3,  # Dysprosium
    "Ho": 3,  # Holmium
    "Er": 3,  # Erbium
    "Tm": 3,  # Thulium
    "Yb": 3,  # Ytterbium
    "Lu": 3,  # Lutetium

    # Actinides
    "Ac": 3,  # Actinium
    "Th": 4,  # Thorium
    "Pa": 5,  # Protactinium
    "U": 6,   # Uranium
    "Np": 7,  # Neptunium
    "Pu": 7,  # Plutonium
    "Am": 7,  # Americium
    "Cm": 7,  # Curium
    "Bk": 7,  # Berkelium
    "Cf": 7,  # Californium
    "Es": 7,  # Einsteinium
    "Fm": 7,  # Fermium
    "Md": 7,  # Mendelevium
    "No": 7,  # Nobelium
    "Lr": 7,  # Lawrencium
}

# Predefined experiments with descriptions and fun facts
experiments = {
    "water_formation": {
        "name": "Formation of Water",
        "description": "Mix Hydrogen (H) and Oxygen (O) to form Water (H2O).",
        "fun_fact": "Water covers about 71% of the Earth's surface and is essential for all known forms of life.",
        "reactants": ["H", "O"],
        "products": ["H2O"],
    },
    "carbon_dioxide_formation": {
        "name": "Formation of Carbon Dioxide",
        "description": "Mix Carbon (C) and Oxygen (O) to form Carbon Dioxide (CO2).",
        "fun_fact": "Carbon dioxide is a greenhouse gas that helps regulate Earth's temperature.",
        "reactants": ["C", "O"],
        "products": ["CO2"],
    },
    "sodium_chloride_formation": {
        "name": "Formation of Sodium Chloride",
        "description": "Mix Sodium (Na) and Chlorine (Cl) to form Sodium Chloride (NaCl).",
        "fun_fact": "Sodium chloride, commonly known as table salt, is essential for human nutrition.",
        "reactants": ["Na", "Cl"],
        "products": ["NaCl"],
    },
    "ammonia_formation": {
        "name": "Formation of Ammonia",
        "description": "Mix Nitrogen (N) and Hydrogen (H) to form Ammonia (NH3).",
        "fun_fact": "Ammonia is widely used in fertilizers and household cleaning products.",
        "reactants": ["N", "H"],
        "products": ["NH3"],
    },
    "methane_formation": {
        "name": "Formation of Methane",
        "description": "Mix Carbon (C) and Hydrogen (H) to form Methane (CH4).",
        "fun_fact": "Methane is the primary component of natural gas and a potent greenhouse gas.",
        "reactants": ["C", "H"],
        "products": ["CH4"],
    },
    "rust_formation": {
        "name": "Formation of Rust",
        "description": "Mix Iron (Fe) and Oxygen (O) to form Rust (Fe2O3).",
        "fun_fact": "Rust is a common example of corrosion and weakens iron structures over time.",
        "reactants": ["Fe", "O"],
        "products": ["Fe2O3"],
    },
    "sulfuric_acid_formation": {
        "name": "Formation of Sulfuric Acid",
        "description": "Mix Sulfur (S), Oxygen (O), and Water (H2O) to form Sulfuric Acid (H2SO4).",
        "fun_fact": "Sulfuric acid is one of the most widely used chemicals in industry.",
        "reactants": ["S", "O", "H2O"],
        "products": ["H2SO4"],
    },
    "glucose_formation": {
        "name": "Formation of Glucose",
        "description": "Mix Carbon (C), Hydrogen (H), and Oxygen (O) to form Glucose (C6H12O6).",
        "fun_fact": "Glucose is the primary source of energy for most living organisms.",
        "reactants": ["C", "H", "O"],
        "products": ["C6H12O6"],
    },
}



# Initialize the text generation model using GPT-J


# Load model and tokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Use GPT-Neo for better accuracy
model_name = "EleutherAI/gpt-neo-125M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Define function for chemistry descriptions
def generate_chemistry_description(elements):
    prompt = (f"You mixed {', '.join(elements)}.\n"
              "What is the correct molecular formula of the compound formed?\n"
              "Provide its common name and a short fun fact.")

    # Use pipeline for text generation
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
    response = generator(prompt, max_length=50, num_return_sequences=1)

    # Extract response text and clean output
    text = response[0]["generated_text"].replace(prompt, "").strip()
    
    return text
    



# Free Lab: All elements in the periodic table
periodic_table = list(valency.keys())

# Homepage
@app.route('/')
def index():
    return render_template('index.html', experiments=experiments)

# Experiment page
@app.route('/experiment/<experiment_id>', methods=['GET', 'POST'])
def experiment(experiment_id):
    experiment_data = experiments.get(experiment_id)
    if not experiment_data:
        return "Experiment not found!", 404

    result = None
    if request.method == 'POST':
        # Get selected reactants from the form
        selected_reactants = request.form.getlist("reactants")
        # Check if the selected reactants match the experiment
        if sorted(selected_reactants) == sorted(experiment_data["reactants"]):
            result = {
                "success": True,
                "message": f"Experiment successful! You created: {', '.join(experiment_data['products'])}.",
                "fun_fact": experiment_data["fun_fact"],
            }
        else:
            result = {
                "success": False,
                "message": "Experiment failed! Incorrect reactants.",
            }

    return render_template('experiment.html', experiment=experiment_data, result=result)

# Free Lab page
@app.route('/free-lab')
def free_lab():
    return render_template('free_lab.html', periodic_table=periodic_table)

# Handle mixing in Free Lab
from math import gcd
from functools import reduce

def lcm(a, b):
    """Calculate the least common multiple of two numbers."""
    return a * b // gcd(a, b)

def lcm_of_list(numbers):
    """Calculate the LCM of a list of numbers."""
    return reduce(lcm, numbers, 1)

@app.route('/mix-elements', methods=['POST'])
def mix_elements():
    data = request.json
    elements = data.get("elements", [])

    # Convert element names to symbols
    symbols = [element.strip() for element in elements]

    # Get valencies of the elements
    valencies = [valency.get(element, 0) for element in symbols]

    # If any element has no valency (e.g., noble gases), return an error
    if 0 in valencies:
        return jsonify({
            "success": False,
            "message": "One or more elements cannot form compounds.",
        })

    # Calculate the LCM of the valencies
    total_lcm = lcm_of_list(valencies)

    # Determine the number of atoms for each element
    atom_counts = [total_lcm // v for v in valencies]

    # Generate reactants side of the equation
    reactants = " + ".join([f"{atom_counts[i]}{symbols[i]}" for i in range(len(symbols))])

    # Generate products side of the equation (simple combination)
    product = "".join([f"{symbols[i]}{atom_counts[i]}" for i in range(len(symbols))])

    # Generate the equation
    equation = f"{reactants} → {product}"

    # Generate a description using the Hugging Face model
    description = generate_chemistry_description(symbols) if symbols else "No valid reaction description available."


    return jsonify({
        "success": True,
        "message": f"You created: {product}!",
        "equation": equation,
        "description": description,
    })


if __name__ == '__main__':
    app.run(debug=True)