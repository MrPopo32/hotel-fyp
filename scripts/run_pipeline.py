"""
Run the main project pipeline in order.
"""

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


def parseArgs() :
	parser = argparse.ArgumentParser()
	parser.add_argument("--skip-extract", action = "store_true")
	parser.add_argument("--skip-vectors", action = "store_true")
	parser.add_argument("--skip-dataset", action = "store_true")
	parser.add_argument("--final-step", choices = ["none", "recommend", "eval"], default = "none")
	parser.add_argument("final_args", nargs = argparse.REMAINDER)
	return parser.parse_args()


def runStep(scriptName, stepLabel, extraArgs = None) :
	extraArgs = extraArgs if extraArgs is not None else []
	scriptPath = SCRIPTS_DIR / scriptName

	command = [sys.executable, str(scriptPath)]
	for value in extraArgs :
		command.append(value)

	print("\n")
	print("Running step: " + stepLabel)
	print("Command: " + " ".join(command))
	print()

	subprocess.run(command, check = True, cwd = str(PROJECT_ROOT))


def getFinalScriptName(finalStep) :
	if finalStep == "recommend":
		return "run_recommendations.py"
	if finalStep == "eval":
		return "run_eval.py"
	return None


def main() :
	args = parseArgs()

	# Each stage can be skipped for partial reruns.
	if not args.skip_extract:
		runStep("extract_sql_to_parquet.py", "Extract SQL to parquet")

	if not args.skip_vectors:
		runStep("build_vectors.py", "Build sentiment vectors")

	if not args.skip_dataset:
		runStep("build_dataset.py", "Build hotel profile dataset")

	finalScriptName = getFinalScriptName(args.final_step)
	if finalScriptName is not None:
		finalArgs = list(args.final_args)

		# Drop argparse's separator before forwarding final-step flags.
		if len(finalArgs) > 0 and finalArgs[0] == "--":
			finalArgs = finalArgs[1:]

		finalLabel = "Run recommendations" if args.final_step == "recommend" else "Run evaluation"
		runStep(finalScriptName, finalLabel, finalArgs)


if __name__ == "__main__":
	main()
