#pragma GCC diagnostic push												
#pragma GCC diagnostic ignored "-Wsign-compare"



// Libs
#include <iostream>
#include <chrono>
#include <fstream>				
#include <vector>
#include <string>
#include <numeric>
#include <algorithm>			
#include <cmath>				
#include <math.h>				

// Namespace
using namespace std;
using namespace std::chrono;

// Start index for test data
const int startTest = 900;	

// Additional control variables
double calcMean(vector<double> vect);
double calcVariance(vector<double> vect);

// Auxiliary methods
void print2DVector(vector < vector <double> > vect);
vector<vector<double> > priorProb(vector < double > vect);
vector<vector<double> > countclassVect(vector<double> vect);
vector<vector<double> > likelihoodtype_doc (vector<double> classVect, vector<double> type_doc, vector<vector<double> > count_classVect);
vector<vector<double> > likelihoodvalid_cert (vector<double> classVect, vector<double> valid_cert, vector<vector<double> > count_classVect);
vector<vector<double> > used_daysMean (vector<double> classVect, vector<double> used_days, vector<vector<double> > count_classVect);
vector<vector<double> > used_daysVar (vector<double> classVect, vector<double> used_days, vector<vector<double> > count_classVect);
vector<vector<double> > used_days_metrics (vector<vector<double> > used_daysMean, vector<vector<double> > used_daysVar);
double calc_used_dayslh (double v, double mean_v, double var_v);

// Method to implement Bayes' Theorem
vector<vector<double> > calc_raw_prob(double type_doc, double valid_cert, double used_days, vector<vector<double> > apriori, vector<vector<double> > lh_type_doc, vector<vector<double> > lh_valid_cert, vector<vector<double> > used_days_mean, vector<vector<double> > used_days_var);

//  Calculation of model evaluation metrics
vector<vector<double> > confusionMatrix(vector<double> matA, vector<double> matB);
double accuracy(vector<double> matA, vector<double> matB);

const int numOfIterations = 5;	

int main() {

    // Define the file name as a string
	string fileName = "data/dataset.csv";

    // Object to receive the content of the file		
	ifstream inputFile;					
	
    // Open the file
	inputFile.open(fileName);	
	
	// Check for any errors
	if(!inputFile.is_open()) {
		cout << "Failed to open the file." << endl;
		return 0;
	}

    // Variable declaration

    // Scalar double variables to handle the values of each column
	double idVal;
	double type_docVal;
	double classVectVal;
	double valid_certVal;
	double used_daysVal;
    
    // Vector variables for all elements of each column of the dataset
	vector<double> id;
	vector<double> type_doc;			
	vector<double> classVect;		
	vector<double> valid_cert;		
	vector<double> used_days;		
	
    // Variable to store the file header
	string header;		

    // Variable to store each cell of the csv file		
	string cell; 	
	
	// Retrieve the header to disregard the first line
	getline(inputFile, header);
	
    // Initial data loading and cleaning loop
    while(inputFile.good()) {
				
        // Read the id column
		getline(inputFile, cell, ','); 			

        // Remove quotes		
		cell.erase(remove(cell.begin(), cell.end(), '\"' ),cell.end());	
	
        // Now we continue reading only the cells with values
		if(!cell.empty()) {								

            // Convert the id from string to double	
			idVal = stod(cell);	

            // Append the value of x to the vector		
			id.push_back(idVal);					
							
            // Read the type_doc column
			getline(inputFile, cell, ','); 	

            // Convert string to double				
			type_docVal = stod(cell);	

            // Append to the vector		
			type_doc.push_back(type_docVal);		
				
            // Read the classVect column	
			getline(inputFile, cell, ',');	

            // Convert to double			
			classVectVal = stod(cell);	

            // Append to the vector		
			classVect.push_back(classVectVal);	
			
            // Read the valid_cert column 
			getline(inputFile, cell, ',');		

            // Convert to double		
			valid_certVal = stod(cell);	

            // Append to the vector				
			valid_cert.push_back(valid_certVal);				
				
            // Read the used_days column
			getline(inputFile, cell);	

            // Convert to double			
			used_daysVal = stod(cell);		

            // Append to the vector				
			used_days.push_back(used_daysVal);					
		}
		else {

            // If the line is empty, finish the loop											

			break;
		}	
	}

	// Start measuring execution time
	auto start = high_resolution_clock::now();  

	cout << "Starting Algorithm Execution" << endl << endl;
	
	// We divide the data creating the training vectors

	// Vector with training data for type_doc
	vector<double> type_doctrain_data;

	// Load the vector
	for(int i = 0; i < startTest; i++) {
		type_doctrain_data.push_back(type_doc.at(i));
	}
	
	// Vector with training data for classVect
	vector<double> classVecttrain_data;

	// Load the vector
	for(int i = 0; i < startTest; i++) {
		classVecttrain_data.push_back(classVect.at(i));
	}
	
	// Vector with training data for valid_cert
	vector<double> valid_certtrain_data;

	// Load the vector
	for(int i = 0; i < startTest; i++) {
		valid_certtrain_data.push_back(valid_cert.at(i));
	}	
	
	// Vector with training data for used_days
	vector<double> used_daystrain_data;
	
	// Load the vector
	for(int i = 0; i < startTest; i++) {
		used_daystrain_data.push_back(used_days.at(i));
	}

	// We divide the data creating the test vectors

	// Vector with test data for type_doc
	vector<double> type_doctest_data;
	
	// Load the vector
	for(int i = startTest; i < id.size(); i++) {
		type_doctest_data.push_back(type_doc.at(i));
	}
	
	// Vector with test data for classVect
	vector<double> classVecttest_data;
	
	// Load the vector
	for(int i = startTest; i < id.size(); i++) {
		classVecttest_data.push_back(classVect.at(i));
	}
	
	// Vector with test data for valid_cert
	vector<double> valid_certtest_data;
	
	// Load the vector
	for(int i = startTest; i < id.size(); i++) {
		valid_certtest_data.push_back(valid_cert.at(i));
	}	
	
	// Vector with test data for used_days
	vector<double> used_daystest_data;
	
	// Load the vector
	for(int i = startTest; i < id.size(); i++) {
		used_daystest_data.push_back(used_days.at(i));
	}



	// MATRIX 1x2 
	vector<vector<double> > apriori = priorProb(classVecttrain_data);					
	cout << "Probabilities " << endl;
	print2DVector(apriori);
	cout << endl;
	
	
	// MATRIX 1x2 
	vector<vector<double> > count_classVect = countclassVect(classVecttrain_data);		
	
	cout << "Conditional probability: " << endl;
	
	// Probability of type_doc variable 
	// MATRIX 2x3
	vector<vector<double> > lh_type_doc = likelihoodtype_doc(classVecttrain_data, type_doctrain_data, count_classVect); 
	cout << "\ttype_doc " << endl;
	print2DVector(lh_type_doc);
	cout << endl;
	
	// Probability of valid_cert variable 
	// MATRIX 2x2
	vector<vector<double> > lh_valid_cert = likelihoodvalid_cert(classVecttrain_data, valid_certtrain_data, count_classVect); 
	cout << "\tvalid_cert: " << endl;
	print2DVector(lh_valid_cert);
	cout << endl;
	
	// Mean and var of used_days variable
	// MATRIX 1x2
	vector<vector<double> > used_days_mean = used_daysMean(classVecttrain_data, used_daystrain_data, count_classVect);	
	vector<vector<double> > used_days_var = used_daysVar(classVecttrain_data, used_daystrain_data, count_classVect);     

	cout << "\tused_days: " << endl;
	vector<vector<double> > used_daysMetrics = used_days_metrics(used_days_mean, used_days_var);
	print2DVector(used_daysMetrics);
	cout << endl << endl;

	cout << "used_days Mean: " << endl;
	print2DVector(used_days_mean);
	cout << endl;

	cout << "used_days Variance: " << endl;
	print2DVector(used_days_var);
	cout << endl << endl;
	
	
	auto stop = high_resolution_clock::now();	

	// 1X2
	vector<vector<double> > raw(1, vector<double> (2, 0)); 
		
	cout << "Predicting probabilities (data test)" << endl;
	

	for(int i = startTest; i < (startTest + numOfIterations); i++) {	

		// 1X2 	
		raw = calc_raw_prob(type_doc.at(i), valid_cert.at(i), used_days.at(i), apriori, lh_type_doc, lh_valid_cert, used_days_mean, used_days_var);   
		print2DVector(raw);
	}
	cout << endl << endl;
	
	std::chrono::duration<double> elapsed_sec = stop-start;		
	cout << "Time spent -> " << elapsed_sec.count() << endl << endl;	

	// Normalize probabilities
	vector<double> p1(146); 
	for(int i = 0; i < type_doctest_data.size(); i++) {

		// 1X2
		raw = calc_raw_prob(type_doctest_data.at(i), valid_certtest_data.at(i), used_daystest_data.at(i), apriori, lh_type_doc, lh_valid_cert, used_days_mean, used_days_var);   
		if((raw.at(0).at(0)) > 0.5 ) {
			p1.at(i) = 0;
		}
		else if((raw.at(0).at(1)) > 0.5) {
			p1.at(i) = 1;
		}
		else {}
	}


	cout << "Confusion Matrix: " << endl;
	vector<vector<double> > table = confusionMatrix(p1, classVecttest_data);
	print2DVector(table); 
	cout << endl;
	
	double acc = accuracy(p1, classVecttest_data);
	cout << "Accuracy: " << acc << endl;
	
	// Sensitivity = TP / (TP + FN)
	double sensitivity = (table.at(0).at(0) / ( table.at(0).at(0) + table.at(1).at(0)));
	cout << "Sensitivity: " << sensitivity << endl;
	
	// Specificity = TN / (TN + FP)
	double specificity = (table.at(1).at(1) / ( table.at(1).at(1) + table.at(0).at(1)));
	cout << "Specificity: " << specificity << endl << endl;
	
	return 0;


}

void print2DVector(vector<vector<double> > vect) {
	for(int i = 0; i < vect.size(); i++) {
		for(int j = 0; j < vect[i].size(); j++) {
			cout << vect[i][j] << " ";
		}
		cout << endl;
	}
}

// Calculte apriori probability (train data) 
vector<vector<double> > priorProb(vector<double> vect) {

	// 1X2
	vector<vector<double> > prior(1, vector<double> (2, 0)); 
	
	for(int i = 0; i < vect.size(); i++) {
		if(vect.at(i) == 0) {
			prior.at(0).at(0)++;
		}
		else {
			prior.at(0).at(1)++;
		}
	}

	prior.at(0).at(0) = prior.at(0).at(0) / vect.size();
	prior.at(0).at(1) = prior.at(0).at(1) / vect.size();
		
	return prior;
}

// Calculate the classVect count 
vector<vector<double> > countclassVect(vector<double> vect) {

	// 1X2
	vector<vector<double> > prior(1, vector<double> (2, 0)); 
	
	for(int i = 0; i < vect.size(); i++) {
		if(vect.at(i) == 0) {
			prior.at(0).at(0)++;
		}
		else {
			prior.at(0).at(1)++;
		}
	}
	return prior;
}

// Calculate the probability of type_doc variable (train data) 
vector<vector<double> > likelihoodtype_doc (vector<double> classVect, vector<double> type_doc, vector<vector<double> > count_classVect) {
	
	// 2X3
	vector<vector<double> > lh_type_doc (2, vector<double>(3,0)); 				
	
	for(int i = 0; i < classVect.size(); i++) {
		if(classVect.at(i) == 0) {
			if(type_doc.at(i) == 1) {
				lh_type_doc.at(0).at(0)++;
			}
			else if(type_doc.at(i) == 2) {
				lh_type_doc.at(0).at(1)++;
			}
			else if (type_doc.at(i) == 3) {
				lh_type_doc.at(0).at(2)++;
			}
			else {}
		}
		else if(classVect.at(i) == 1) {
			if(type_doc.at(i) == 1) {
				lh_type_doc.at(1).at(0)++;
			}
			else if(type_doc.at(i) == 2) {
				lh_type_doc.at(1).at(1)++;
			}
			else if (type_doc.at(i) == 3) {
				lh_type_doc.at(1).at(2)++;
			}
			else {}
		}
		else{}
	}
	
	for(int i = 0; i < lh_type_doc.size(); i++) {
		for(int j = 0; j < lh_type_doc[i].size(); j++) {
			if(i == 0) {
				lh_type_doc.at(i).at(j) = lh_type_doc.at(i).at(j) / count_classVect.at(0).at(0);
			}
			else if(i == 1) {
				lh_type_doc.at(i).at(j) = lh_type_doc.at(i).at(j) / count_classVect.at(0).at(1);
			}
		}
	}
	
	return lh_type_doc;
}

// Calculate the probability of the valid_cert variable (train data) 
vector<vector<double> > likelihoodvalid_cert (vector<double> classVect, vector<double> valid_cert, vector<vector<double> > count_classVect) {
	
	// 2X2
	vector<vector<double> > lh_valid_cert (2, vector<double>(2,0)); 			
	
	for(int i = 0; i < classVect.size(); i++) {
		if(classVect.at(i) == 0) {
			if(valid_cert.at(i) == 0) {
				lh_valid_cert.at(0).at(0)++;
			}
			else if(valid_cert.at(i) == 1) {
				lh_valid_cert.at(0).at(1)++;
			}
			else {}
		}
		else if(classVect.at(i) == 1) {
			if(valid_cert.at(i) == 0) {
				lh_valid_cert.at(1).at(0)++;
			}
			else if(valid_cert.at(i) == 1) {
				lh_valid_cert.at(1).at(1)++;
			}
			else {}
		}
		else{}
	}

	for(int i = 0; i < lh_valid_cert.size(); i++) {
		for(int j = 0; j < lh_valid_cert[i].size(); j++) {
			if(i == 0) {
				lh_valid_cert.at(i).at(j) = lh_valid_cert.at(i).at(j) / count_classVect.at(0).at(0);
			}
			else if(i == 1) {
				lh_valid_cert.at(i).at(j) = lh_valid_cert.at(i).at(j) / count_classVect.at(0).at(1);
			}
		}
	}
	
	return lh_valid_cert;
}

// Calculate the mean of used_days variable 
vector<vector<double> > used_daysMean (vector<double> classVect, vector<double> used_days, vector<vector<double> > count_classVect) {

	// 1X2 MATRIX
	vector<vector<double> > mean(1, vector<double> (2, 0));
		
	for(int i = 0; i < classVect.size(); i++) {
		if(classVect.at(i) == 0) {
			mean.at(0).at(0) += used_days.at(i);
		}
		else if(classVect.at(i) == 1) {
			mean.at(0).at(1) += used_days.at(i);
		}
		else{}
	}
		
	for(int i = 0; i < mean.size(); i++) {
		for(int j = 0; j < mean[i].size(); j++) {
			if(j == 0) {
				mean.at(i).at(j) = mean.at(i).at(j) / count_classVect.at(0).at(0);
			}
			else if(j == 1) {
				mean.at(i).at(j) = mean.at(i).at(j) / count_classVect.at(0).at(1);
			}
		}
	}
	return mean;
}

// Calculate the variance of used_days variable 
vector<vector<double> > used_daysVar (vector<double> classVect, vector<double> used_days, vector<vector<double> > count_classVect) {

	// 1X2 MATRIX
	vector<vector<double> > var(1, vector<double> (2, 0)); 

	// 1X2 MATRIX
	vector<vector<double> > mean = used_daysMean(classVect, used_days, count_classVect); 
			
	for(int i = 0; i < classVect.size(); i++) {
		if(classVect.at(i) == 0) {
			var.at(0).at(0) += pow( ( used_days.at(i) - mean.at(0).at(0) ), 2);
		}
		else if(classVect.at(i) == 1) {
			var.at(0).at(1) += pow( ( used_days.at(i) - mean.at(0).at(1) ), 2);
		}
		else{}
	}
		
	for(int i = 0; i < var.size(); i++) {
		for(int j = 0; j < var[i].size(); j++) {
			if(j == 0) {
				var.at(i).at(j) = var.at(i).at(j) / ( count_classVect.at(0).at(0) - 1 ) ;
			}
			else if (j == 1) {
				var.at(i).at(j) = var.at(i).at(j) / ( count_classVect.at(0).at(1) - 1 ) ;
			}
			else {}
		}
	}	
	return var;
}

vector<vector<double> > used_days_metrics (vector<vector<double> > used_daysMean, vector<vector<double> > used_daysVar) {

	// 2X2 MATRIX
	vector<vector<double> > metrics(2, vector<double>(2, 0));  
	
	metrics.at(0).at(0) = used_daysMean.at(0).at(0);
	metrics.at(0).at(1) = sqrt(used_daysVar.at(0).at(0));
	metrics.at(1).at(0) = used_daysMean.at(0).at(1);
	metrics.at(1).at(1) = sqrt(used_daysVar.at(0).at(1));
	
	return metrics;
}

// Calculate the probability of used_days variable 
double calc_used_dayslh (double v, double mean_v, double var_v) {
	double used_days_lh = 0;
	
	// Probability
	used_days_lh = (1 / (sqrt(2 * M_PI * var_v))) * exp( -(pow((v - mean_v), 2)) / (2 * var_v));
	
	return used_days_lh;
}

// Bayes
vector<vector<double> > calc_raw_prob(double type_doc, double valid_cert, double used_days, vector<vector<double> > apriori, vector<vector<double> > lh_type_doc, vector<vector<double> > lh_valid_cert, vector<vector<double> > used_days_mean, vector<vector<double> > used_days_var) {
	
	// 1x2 MATRIX
	vector<vector<double> > raw(1, vector<double> (2, 0)); 
	
	// Output variable probability
	double num_s = lh_type_doc.at(1).at(type_doc-1) * lh_valid_cert.at(1).at(valid_cert) * apriori.at(0).at(1) *
					calc_used_dayslh(used_days, used_days_mean.at(0).at(1), used_days_var.at(0).at(1));		
					
	// Input variable probability 
	double num_p = lh_type_doc.at(0).at(type_doc-1) * lh_valid_cert.at(0).at(valid_cert) * apriori.at(0).at(0) *
					calc_used_dayslh(used_days, used_days_mean.at(0).at(0), used_days_var.at(0).at(0));				
		
	// Denominator
	double denominator = lh_type_doc.at(1).at(type_doc-1) * lh_valid_cert.at(1).at(valid_cert) *
					calc_used_dayslh(used_days, used_days_mean.at(0).at(1), used_days_var.at(0).at(1)) * apriori.at(0).at(1)
					+ lh_type_doc.at(0).at(type_doc-1) * lh_valid_cert.at(0).at(valid_cert) * 
					calc_used_dayslh(used_days, used_days_mean.at(0).at(0), used_days_var.at(0).at(0)) * apriori.at(0).at(0);
		
	raw.at(0).at(1) = num_s / denominator;
	raw.at(0).at(0) = num_p / denominator;
	
	return raw;
}

vector<vector<double> > confusionMatrix(vector<double> matA, vector<double> matB) {

	// 2x2 MATRIX
	vector<vector<double> > table(2, vector<double>(2, 0));  
	
	
	for(int i = 0; i < matA.size(); i++) {

			// true negative
			if( matA.at(i) == 0 && matB.at(i) == 0 ) {			
				table.at(0).at(0)++;
			}

			// true positive
			else if( matA.at(i) == 1 && matB.at(i) == 1 ) {		
				table.at(1).at(1)++;
			}

			// false positive
			else if( matA.at(i) == 1 && matB.at(i) == 0 ) {	
				table.at(1).at(0)++;
			}

			// false negative
			else if( matA.at(i) == 0 && matB.at(i) == 1 ) {		
				table.at(0).at(1)++;
			}
			else {}
		}	
	return table;
}

// Return accuracy
double accuracy(vector<double> matA, vector<double> matB) {
	int matARow = matA.size();
	int matBRow = matB.size();	
	
	if((matARow != matBRow)) {
		cout << "Erro, dimensions need to be the same!" << endl;
	}
		
	double sum = 0;
	
	for(int i = 0; i < matA.size(); i++) {
		if(matA.at(i) == matB.at(i)) {
				sum++;
		}

	}	
	return sum / matA.size();
}



