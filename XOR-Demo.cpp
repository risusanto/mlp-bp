#include "trainingData-Demo.cpp"

void showVectorVals(string label, vector<double> &v)
{
	cout << label << " ";
	for(unsigned i = 0; i < v.size(); ++i)
	{
		cout << v[i] << " ";
	}
	cout << endl;
}
int main()
{
	TrainingData trainData("data.txt");
	vector<unsigned> topology;

    ofstream file ("output.txt");
	trainData.getTopology(topology);
	Net myNet(topology);

	vector<double> inputVals, targetVals, resultVals;
	int trainingPass = 0;
	while(!trainData.isEof())
	{
		++trainingPass;
		file << "Attempt: " << trainingPass;

		// Get new input data and feed it forward:
		if(trainData.getNextInputs(inputVals) != topology[0])
			break;
        file << " Inputs: " << inputVals.at(inputVals.size()-2) << " " << inputVals.at(inputVals.size()-1) << "\n";
		myNet.feedForward(inputVals);

		// Collect the net's actual results:
		myNet.getResults(resultVals);

		file << "Computer Answer: " << resultVals.back() << "\n";
		// Train the net what the outputs should have been:
		trainData.getTargetOutputs(targetVals);
        file << "Correct Answer: " << targetVals.back() << "\n";
		assert(targetVals.size() == topology.back());

		myNet.backProp(targetVals);

		// Report how well the training is working, average over recent
        file << "Average Error: " << myNet.getRecentAverageError() << "\n\n";
	}

	cout << "PROCESS SUCCESSFUL: output.txt\n";

}
