import os
from dataclasses import asdict, dataclass
from pathlib import Path
import pickle
from typing import *
from typing import List, Tuple

import numpy as np
import openai
import tiktoken
from jinja2 import Template
import json
from textwrap import dedent
from pydantic import BaseModel

from neurons.constants import PRICING_DATA_PER_MILLION_TOKENS
from neurons.helpers import logger

@dataclass
class File:
    path: Path
    contents: str


@dataclass
class EmbeddedFile:
    path: str
    contents: str
    embedding: list

    def __str__(self):
        return f"File: {self.path}, Length: {len(self.contents)}"

    def __repr__(self) -> str:
        return f"File: {self.path}, Length: {len(self.contents)}"


@dataclass
class FilePair:
    cosine_similarity: float
    files: List[EmbeddedFile]

@dataclass
class IngestionHeuristics:
    min_files_to_consider_dir_for_problems: int
    min_file_content_len: int

SAMPLE_INGESTION_HEURISTICS = IngestionHeuristics(
    min_files_to_consider_dir_for_problems=5,
    min_file_content_len=50
)

@dataclass
class ProblemGeneratorParameters:
    filepair_selection_logic: Callable[[List[FilePair]], FilePair]
    prompt_template: Template
    problem_gen_model: str

@dataclass
class ValidatorModelStats:
    input_tokens: int
    output_tokens: int
    cost: float

@dataclass
class GeneratedProblemStatement:
    repo_path: Path
    prompt: str
    model: str
    problem_statement: str
    dynamic_checklist: List[str]
    model_stats: Optional[ValidatorModelStats] = None

class GeneratedProblem(BaseModel):
    problem_statement: str
    dynamic_checklist: List[str]


# We use pydantic for some classes because OpenAI json output can structure based on that
class ListOfGeneratedProblems(BaseModel):
    generated_problem_statements: List[GeneratedProblem]


OPENAI_CLIENT: Final[openai.Client] = openai.Client(api_key=os.getenv("OPENAI_API_KEY"))

PROBLEM_STATEMENT_TEMPLATE: Final[Template] = Template(
    dedent("""
    You are a skilled software engineering assistant. You will be provided with multiple files as context. Each file will contain portions of code, documentation, or relevant information about a software system. Your task is to come up with a specific software engineering problem that requires a solution to involve at least two of these files. You will generate a list of these problems, in the generated_problems array response.
    
    Further, once you have a problem statement, generate a checklist of points to consider and things that should be present in the solution (for example, are the correct Github API calls made if its a function that interfaces with the api). Generate several of these into dynamic_checklist field.
    Some additional guidelines are:
    - Do not output anything other than software engineering problem
    - The problem description should be very detailed and meticulous. It should contain sufficient context such that someone equipped with the codebase and your problem statement will have enough information to implement
    - The problem should be solvable by an autonomous SWE which can do things like installing PyPi packages, but cannot do things like make cloud provider accounts and register for other services manually.
    - The problem should not be overly difficult to implement, and should be fairly easy and not take too many LLM calls. 
    - Do not disclose which files would need to be modified to solve the problem.
    
    Here are the files:
    {% for file in files %}
    Filename: {{ file.path }}
    ```python3
    {{ file.contents }}
    ```
    {% endfor %}
    ```
    """)
)

def walk_repository(repo_path: Path) -> Dict:
    """
    Picks files to generate problem statements from
    """
    repo_map = {}

    for root, dirs, files in os.walk(str(repo_path)):
        # Convert absolute path to relative path from repo root
        rel_path = os.path.relpath(root, str(repo_path))
        if rel_path == '.':
            rel_path = ''

        # Filter out common files/directories to ignore
        dirs[:] = [d for d in dirs if not d.startswith(('.', '__'))]
        files = [f for f in files if not f.startswith(('.', '__')) and not f.endswith(('.pyc', '.pyo')) and f.endswith('.py')]

        # Add to map
        repo_map[rel_path] = {
            'dirs': dirs,
            'files': files
        }

    return repo_map


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def evaluate_for_context(
        dir_path, 
        repo_structure,
        heuristics: IngestionHeuristics
    ):
    def _retrieve_files_in_dir():
        # Get all files in the current directory
        files = []
        for file_name in repo_structure['files']:
            path = os.path.join(dir_path, file_name)
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    contents = f.read()
                files.append(
                    {
                        'path': path,
                        'contents': contents
                    }
                )
            except (UnicodeDecodeError, IOError):
                logger.exception(f"Warning: Could not read file {path}")
                continue

        return files

    def _embed_code(raw_codes: List[str]) -> List[List[float]]:
        encoding = tiktoken.get_encoding('cl100k_base')
        truncated_inputs = [encoding.encode(json.dumps(code))[:8191] for code in raw_codes]
        response = OPENAI_CLIENT.embeddings.create(
            model="text-embedding-3-small",
            input=truncated_inputs
        )
        # Return list of embedding vectors
        return [data.embedding for data in response.data]

    def _find_most_similar_files(embedded_files: List[EmbeddedFile]) -> FilePair | None:
        max_similarity = -1
        most_similar_pair = None

        # Compare each pair of files
        for i in range(len(embedded_files)):
            for j in range(i + 1, len(embedded_files)):
                similarity = cosine_similarity(
                    embedded_files[i].embedding,
                    embedded_files[j].embedding
                )

                if similarity > max_similarity:
                    max_similarity = similarity
                    most_similar_pair = [embedded_files[i], embedded_files[j]]

        if most_similar_pair is None:
            return None

        return FilePair(
            cosine_similarity=max_similarity,
            files=most_similar_pair
        )

    if len(repo_structure['files']) >= heuristics.min_files_to_consider_dir_for_problems:
        files = _retrieve_files_in_dir()
        embeddings = _embed_code(list(map(lambda file: file['contents'], files)))
        embedded_files = [
            EmbeddedFile(
                path=file['path'],
                contents=file['contents'],
                embedding=embeddings[i]
            )
            for i, file in enumerate(files)
            if len(file['contents']) > heuristics.min_file_content_len
        ]

        most_similar_files = _find_most_similar_files(embedded_files)

        return most_similar_files
    else:
        return []

def highest_cosine_filepair_selector(file_pairs: List[FilePair]) -> FilePair:
    selected_file_pair = sorted(
        file_pairs,
        key=lambda x: float(x.cosine_similarity),
        reverse=True
    )[0]

    return selected_file_pair

def load_filepairs_from_cache(cache_path: str) -> List[FilePair]:
    """Load FilePairs from cache, returns empty list if cache doesn't exist."""
    try:
        with open(cache_path, 'rb') as f:
            data = pickle.load(f)
        
        # Reconstruct FilePairs from serialized data
        return [
            FilePair(
                cosine_similarity=pair['cosine_similarity'],
                files=[EmbeddedFile(**f) for f in pair['files']]
            )
            for pair in data
        ]
    except (FileNotFoundError, EOFError):
        return []
    
def save_filepairs_to_cache(filepairs: List[FilePair], cache_path: str) -> None:
    """Save list of FilePairs to local cache."""
    cache_dir = Path(cache_path).parent
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert FilePairs to serializable format
    serializable_pairs = [
        {
            'cosine_similarity': pair.cosine_similarity,
            'files': [asdict(f) for f in pair.files]
        }
        for pair in filepairs
    ]
    
    with open(cache_path, 'wb') as f:
        pickle.dump(serializable_pairs, f)

def get_all_filepairs(
    local_repo: Path, 
    heuristics: IngestionHeuristics = SAMPLE_INGESTION_HEURISTICS,
    refresh: bool = False,
) -> List[FilePair]:
    cache_path = f".cache/{local_repo}"

    filepairs_from_cache = load_filepairs_from_cache(cache_path=cache_path)

    if filepairs_from_cache and len(filepairs_from_cache) > 0 and refresh == False:
        return filepairs_from_cache
    
    repo_structure = walk_repository(local_repo)

    file_pairs = []
    for dir_path, contents in repo_structure.items():
        full_path = os.path.join(local_repo, dir_path) if dir_path else local_repo
        if contents['files']:
            file_pairs.append(evaluate_for_context(full_path, contents, heuristics=heuristics))

    valid_pairs = [pair for pair in file_pairs if pair and isinstance(pair, FilePair)]
    if not valid_pairs:
        raise ValueError("No valid file pairs found in the repository. Ensure there are directories with 5+ Python files.")
    
    # Save the filepairs to cache
    save_filepairs_to_cache(
        filepairs=valid_pairs,
        cache_path=cache_path
    )

    return valid_pairs

def calculate_price(model_name: str, input_tokens: int, output_tokens: int) -> float:
    pricing_dict = PRICING_DATA_PER_MILLION_TOKENS[model_name]
    input_price, output_price = pricing_dict["input"], pricing_dict["output"]
    return (input_tokens * input_price + output_tokens * output_price) / 1e6

def generate_for_single_repo(
        repo_path: Path,
        ingestion_heuristics: IngestionHeuristics,
        params: ProblemGeneratorParameters
) -> GeneratedProblemStatement:
    file_pairs = get_all_filepairs(repo_path, ingestion_heuristics, refresh=False)

    selected_file_pair = params.filepair_selection_logic(file_pairs)
    prompt_text = params.prompt_template.render(
        dict(files=selected_file_pair.files)
    )

    model_map = {
        "gpt4omini": "gpt-4o-mini",
        "gpt4o": "gpt-4o"
    }
    model = params.problem_gen_model if params.problem_gen_model not in model_map \
        else model_map[params.problem_gen_model]
    
    completion = OPENAI_CLIENT.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": prompt_text},
            {"role": "user", "content": f"Generate the list of problem statements. Generate exactly 1 statement, no more and no less"},
        ],
        response_format=ListOfGeneratedProblems,
    )

    parsed_response = completion.choices[0].message.parsed.generated_problem_statements
    prompt_tokens, completion_tokens = completion.usage.prompt_tokens, completion.usage.completion_tokens
    cost = calculate_price(model, prompt_tokens, completion_tokens)

    return [
        GeneratedProblemStatement(
            repo_path=repo_path,
            prompt=prompt_text,
            model=model,
            problem_statement=statement.problem_statement,
            dynamic_checklist=statement.dynamic_checklist,
            model_stats=ValidatorModelStats(
                input_tokens=prompt_tokens,
                output_tokens=completion_tokens,
                cost=cost,
            ),
        ) for statement in parsed_response
    ]


def generate_problem_statement(local_repo: Path) -> GeneratedProblemStatement:
    """
    Generate a problem statement given the repository.

    Args:
        local_repo: Path to the local repository.

    Returns:
        GeneratedProblemStatement: The generated problem statement.
    """
    problem_generator_params = ProblemGeneratorParameters(
        filepair_selection_logic=highest_cosine_filepair_selector,
        prompt_template=PROBLEM_STATEMENT_TEMPLATE,
        problem_gen_model="gpt-4o"
    )

    ingestion_heuristics = IngestionHeuristics(
        min_files_to_consider_dir_for_problems=3,
        min_file_content_len=50,
    )

    problem_statement: GeneratedProblemStatement = generate_for_single_repo(
        repo_path=local_repo,
        ingestion_heuristics=ingestion_heuristics,
        problem_generator_params=problem_generator_params
    )

    return problem_statement[0]
