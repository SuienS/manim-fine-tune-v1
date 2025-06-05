"""code_evaluator.py: Evaluate code snippets by comparing outputs to expected results."""
__author__      = "Ravidu Silva"

from codebleu import calc_codebleu
import ast
from zss import simple_distance, Node
from transformers import RobertaTokenizer, RobertaModel
import torch

import config

class CodeEvaluator:
    """Class to evaluate code snippets by comparing outputs to expected results."""


    def __init__(self, model_name: str = config.SupportedModels.CODE_EVALUATOR_MODEL):
        """
        Initialize the CodeEvaluator with the specified model.

        Args:
            model_name (str): The name of the Hugging Face model to use.
        """
        
        print(f"Loading model: '{model_name}'...")
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.model = RobertaModel.from_pretrained(model_name)
        self.model.eval()

    def evaluate_code(self, generated_code: str, expected_code: str) -> dict:
        """
        Calculate all evaluation scores (CodeBERT Similarity, CodeBLEU, AST Distance) for the generated and expected code.

        Args:
            generated_code (str): The generated code.
            expected_code (str): The expected code.

        Returns:
            dict: A dictionary containing the evaluation scores.
        """

        results = {
            "codebert_similarity": self.codebert_similarity(generated_code, expected_code)
        }
        results.update(self.calculate_codebleu(generated_code, expected_code))
        results.update(self.calculate_ast_distance(generated_code, expected_code))

        return results


    def codebert_similarity(self, code1: str, code2: str) -> float:
        """
        Calculate the similarity between two code snippets using CodeBERT.

        Args:
            code1 (str): The first code snippet.
            code2 (str): The second code snippet.

        Returns:
            float: The similarity score between the two code snippets.
        """

        inputs = self.tokenizer([code1, code2], return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
        similarity = torch.cosine_similarity(embeddings[0], embeddings[1], dim=0).item()

        return similarity


    def calculate_codebleu(self, generated_code: str, expected_code: str) -> dict:
        """
        Calculate the CodeBLEU score between generated and expected code.

        Args:
            generated_code (str): The generated code.
            expected_code (str): The expected code.

        Returns:
            dict: A dictionary containing the CodeBLEU score and its components: 
                - CodeBLEU: The CodeBLEU score.
                - N-gram match: The n-gram match score.
                - Weighted n-gram match: The weighted n-gram match score.
                - Syntax match: The syntax match score.
                - Dataflow match: The dataflow match score.
        """
        return calc_codebleu(
            references=[expected_code],
            predictions=[generated_code],
            lang="python"
        )
    

    def calculate_ast_distance(self, generated_code: str, expected_code: str) -> dict:
        """
        Calculate the AST distance between generated and expected code.

        Args:
            generated_code (str): The generated code.
            expected_code (str): The expected code.

        Returns:
            float: The AST distance.
        """

        try:
            generated_ast = self.__ast_to_node(ast.parse(generated_code))
            syntax_error = False
        except SyntaxError as e:
            # Not a valid Python code
            syntax_error = True
            return {
                "ast_distance_norm": None,
                "ast_distance_raw": None,
                "ast_distance_max": None,
                "syntax_error": syntax_error
            }

        expected_ast = self.__ast_to_node(ast.parse(expected_code))

        # Calculate the number of nodes in each AST
        num_nodes_generated = self.__count_nodes(generated_ast)
        num_nodes_expected = self.__count_nodes(expected_ast)

        # Calculate the maximum possible distance (total nodes in the larger tree)
        max_distance = max(num_nodes_generated, num_nodes_expected)
        
        distance = simple_distance(generated_ast, expected_ast)
        normalized_distance = distance /( max_distance + 1e-7 ) # Avoid division by zero

        return {
            "ast_distance_norm": normalized_distance,
            "ast_distance_raw": distance,
            "ast_distance_max": max_distance,
            "syntax_error": syntax_error
        }

    @staticmethod
    def __ast_to_node(ast_obj):
        """
        Convert an AST object into a tree of Node objects for comparison.

        Args:
            ast_obj: The AST object to convert.

        Returns:
            Node: The root node of the converted tree.
        """
        if isinstance(ast_obj, ast.AST):
            node = Node(type(ast_obj).__name__)
            for field, value in ast.iter_fields(ast_obj):
                if isinstance(value, list):
                    for item in value:
                        node.addkid(CodeEvaluator.__ast_to_node(item))
                elif isinstance(value, ast.AST):
                    node.addkid(CodeEvaluator.__ast_to_node(value))
                else:
                    node.addkid(Node(f"{field}: {value}"))
            return node
        else:
            return Node(str(ast_obj))
        

    @staticmethod
    def __count_nodes(node: Node) -> int:
        """
        Count the number of nodes in a tree of Node objects.

        Args:
            node (Node): The root node of the tree.

        Returns:
            int: The total number of nodes in the tree.
        """
        if not node.children:
            return 1
        return 1 + sum(CodeEvaluator.__count_nodes(child) for child in node.children)