import json
from typing import Any

from sample_usage import Inference


class Get_Input_and_Check_Output:
    def __init__(self, file_name: str) -> None:
        with open(file_name, "r") as file:
            self.data = json.load(file)

    def get_output(self) -> None:
        n = len(self.data)
        output: list[dict[str, Any]] = []
        for i in range(n):
            inference = Inference(self.data[i]["Input"])
            inference.triangulate_and_get_cliques()
            inference.get_junction_tree()
            inference.assign_potentials_to_cliques()
            z_value = inference.get_z_value()
            marginals = inference.compute_marginals()
            top_k_assignments = inference.compute_top_k()
            given = self.data[i]["Output"]
            calc: dict[str, Any] = {
                "Marginals": marginals,
                "Top_k_assignments": top_k_assignments,
                "Z_value": z_value,
            }
            assert given == calc
            output.append(calc)
        self.output = output

    def write_output(self, file_name: str) -> None:
        with open(file_name, "w") as file:
            json.dump(self.output, file, indent=4)


if __name__ == "__main__":
    evaluator = Get_Input_and_Check_Output("testcases/TestCases.json")
    evaluator.get_output()
    evaluator.write_output("testcases/Sample_Testcase_Output.json")
