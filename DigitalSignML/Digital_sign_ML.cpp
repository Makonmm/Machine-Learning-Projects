#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-compare"


// Libs

#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>
#include <string>
#include <numeric>
#include <algorithm>
#include <math.h>
#include <cmath>


// Namespace

using namespace std;
using namespace std::chrono;



const int startTest = 836;

double calcMean(vector<double> vect);
double calcVariance(vector<double> vect);


// Methods

void print2DVector(vector <vector <double> > vect);
vector<vector<double> > priorProb(vector <double> vect);
vector<vector<double> > countclass_(vector <double> vect);
vector<vector<double> > likelihoodtype_doc(vector<double> class_, vector<double> type_doc, vector<vector<double> > count_class);
vector<vector<double> > likelihoodvalid_cert(vector<double> class_, vector<double> valid_cert, vector<vector<double> > count_class);
vector<vector<double> > used_daysMean(vector<double> class_, vector<double> used_days, vector<vector<double> > count_class);
vector<vector<double> > used_daysVar(vector<double> class_, vector<double> used_days, vector<vector<double> > count_class);
vector<vector<double> > used_days_metrics (vector<vector<double> > used_daysMean, vector<vector<double> > used_daysVar);
double calc_used_days_lh(double v, double mean_v, double var_v);


vector<vector<double> > calc_raw_prob(double type_doc, double valid_cert, double used_days, vector<vector<double> > apriori, vector<vector<double> > lh_type_doc, vector<vector<double> > lh_valid_cert, vector<vector<double> > used_days_mean,vector<vector<double> > used_days_var);
vector<vector<double> > confusionMatrix(vector<double> matA, vector<double> matB);
double accuracy(vector<double> matA, vector<double> matB);

const int numOfIterations = 5;



int main() {

    string fileName = "data/dataset.csv";

    // Object that receives the archive content
    ifstream inputFile;

    // Open the archive

    inputFile.open(fileName);

    // Verify errors

    if(!inputFile.is_open()) {
        cout << "Error while trying to open the archive." << endl;
        return 0;
    }

    // Variables

    double type_docVal;
    double idVal;
    double classVal;
    double valid_certVal;
    double used_daysVal;


    vector<double> id;
    vector<double> type_doc;
    vector<double> class_;
    vector<double> valid_cert;
    vector<double> used_days;

    string header;
    string cell;

    getline(inputFile, header);

    while(inputFile.good()) {

        // Reading id column

        getline(inputFile, cell, ',');

        cell.erase(remove(cell.begin(), cell.end(), '\"'), cell.end());

        if(!cell.empty()) {

            // Converting id type string to type double

            idVal = stod(cell);

            // Append the x value in vector

            id.push_back(idVal);

            // Reading type_doc column

            getline(inputFile, cell, ',');

            // Converting type_doc to double

            type_docVal = stod(cell);

            // Append in vector

            type_doc.push_back(type_docVal);

            // Reading class_ column

            getline(inputFile, cell, ',');

            // Converting class_ column to double

            classVal = stod(cell);
            // Append to vector

            class_.push_back(classVal);

            // Reading valid_cert column

            getline(inputFile, cell, ',');

            // Converting valid_cert column to double

            valid_certVal = stod(cell);

            // Append to vector

            valid_cert.push_back(valid_certVal);

            // Reading used_days column

            getline(inputFile, cell, ',');

            // Converting used_days to double

            used_daysVal = stod(cell);

            // Append to vector


            used_days.push_back(used_daysVal);
        }
        else {

            break;
        }
    }

    // Splitting data to train

    auto start = high_resolution_clock::now();

    cout << "Initializing..." << endl << endl;

    vector<double> type_doctrain_data;

    for(int i = 0; i < startTest; i++) {
        type_doctrain_data.push_back(type_doc.at(i));

    }

    vector<double> class_train_data;

    for(int i = 0; i < startTest; i++) {
        class_train_data.push_back(class_.at(i));
    }


    vector<double> valid_certtrain_data;

    for(int i = 0; i < startTest; i++) {
        valid_certtrain_data.push_back(valid_cert.at(i));
    }

    vector<double> used_daystrain_data;

    for(int i = 0; i < startTest; i++){
        used_daystrain_data.push_back(used_days.at(i));
    }

    vector<double> type_doctest_data;

    for(int i = startTest; i < id.size(); i++) {
        type_doctest_data.push_back(type_doc.at(i));
    }

    vector<double> class_test_data;

    for(int i = startTest; i < id.size(); i++) {
        class_test_data.push_back(class_.at(i));
    }

    vector<double> valid_certtest_data;

    for(int i = startTest; i < id.size(); i++) {
        valid_certtest_data.push_back(valid_cert.at(i));
    }

    vector<double> used_daystest_data;
    
    for(int i = startTest; i < id.size(); i++) {
        used_daystest_data.push_back(used_days.at(i));
    }

    // Naive Bayes algorithm

    cout << "Probability: " << endl;

    vector<vector<double> > apriori = priorProb(class_train_data);
    cout << "Probability: " << endl;

    print2DVector(apriori);

    cout << endl;

    vector<vector<double> > count_class = countclass_(class_train_data);

    cout << "Conditional probability: " << endl;

    vector<vector<double> > lh_type_doc = likelihoodtype_doc(class_train_data, type_doctrain_data, count_class);
    
    cout << "\ttype_doc" << endl;
    print2DVector(lh_type_doc);
    cout << endl;

    vector<vector<double> > lh_valid_cert = likelihoodvalid_cert(class_train_data, valid_certtrain_data, count_class);
    
    cout << "\tvalid_cert: " << endl;
    print2DVector(lh_valid_cert);
    cout << endl;

    vector<vector<double> > used_days_mean = used_daysMean(class_train_data, used_daystrain_data, count_class);
    vector<vector<double> > used_days_var = used_daysVar(class_train_data, used_daystrain_data, count_class);

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

    vector<vector<double>> raw(1, vector<double>(2, 0));

    auto stop = high_resolution_clock::now();

    cout << "Predicting probability (data test)" << endl;

    for(int i = startTest; i < (startTest + numOfIterations); i++) {

        raw = calc_raw_prob(type_doc.at(i), valid_cert.at(i), used_days.at(i), apriori, lh_type_doc, lh_valid_cert, used_days_mean, used_days_var);
        print2DVector(raw);

    }

    cout << endl << endl;

    std::chrono::duration<double> elapsed_sec = stop-start;		
	cout << "Execution time: " << elapsed_sec.count() << endl << endl;

    vector<double> p1(146);
    
    for(int i = 0; i < type_doctest_data.size(); i++) {
        raw = calc_raw_prob(type_doctest_data.at(i), valid_certtest_data.at(i), used_daystest_data.at(i), apriori, lh_type_doc, lh_valid_cert, used_days_mean, used_days_var);

        if((raw.at(0).at(0)) > 0.5) {
            p1.at(i) = 0;
        }
        else if((raw.at(0).at(1)) > 0.5) {
            p1.at(i) = 1;
        }
        else {}
    }

    cout << "Confusion matrix: " << endl;
    
    vector<vector<double> > table = confusionMatrix(p1, class_test_data);
    print2DVector(table);
    cout << endl;

    double acc = accuracy(p1, class_test_data);
    cout << "Accuracy: " << acc << endl;

    double sensitivity = (table.at(0).at(0) / (table.at(0).at(0) + table.at(1).at(0)));
    cout << "Sensitivity -> " << sensitivity << endl;

    double specificity = (table.at(1).at(1) / (table.at(1).at(1) + table.at(0).at(1)));
    cout << "Specificty -> " << specificity << endl; 



}

// Method that prints the vector

void print2DVector(vector<vector<double> > vect) {
    for(int i = 0; i < vect.size(); i++) {
        for(int j = 0; j < vect[i].size(); j++) {
            cout << vect[i][j] << " ";
        }
        cout << endl;
    }
}

// Method that calculate apriori probability

vector<vector<double> > priorProb(vector<double> vect) {

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

// Method that counts class_s ocurrencies

vector<vector<double> > countclass_(vector<double> vect) {

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

vector<vector<double> > likelihoodtype_doc(vector<double> class_, vector<double> type_doc, vector<vector<double> > count_class) {
    vector<vector<double> > lh_type_doc(2, vector<double>(3, 0));

    for(int i = 0; i < class_.size(); i++) {
        if(class_.at(i) == 0) {
            if(type_doc.at(i) == 1) {
                lh_type_doc.at(0).at(0)++;
            } else if(type_doc.at(i) == 2) {
                lh_type_doc.at(0).at(1)++;
            } else if(type_doc.at(i) == 3) {
                lh_type_doc.at(0).at(2)++;
            }
        } else if(class_.at(i) == 1) {
            if(type_doc.at(i) == 1) {
                lh_type_doc.at(1).at(0)++;
            } else if(type_doc.at(i) == 2) {
                lh_type_doc.at(1).at(1)++;
            } else if(type_doc.at(i) == 3) {
                lh_type_doc.at(1).at(2)++;
            }
        }
    }
    for(int i = 0; i < lh_type_doc.size(); i++) {
		for(int j = 0; j < lh_type_doc[i].size(); j++) {
			if(i == 0) {
				lh_type_doc.at(i).at(j) = lh_type_doc.at(i).at(j) / count_class.at(0).at(0);
			}
			else if(i == 1) {
				lh_type_doc.at(i).at(j) = lh_type_doc.at(i).at(j) / count_class.at(0).at(1);
			}
		}
	}
    return lh_type_doc;
}



vector<vector<double> > likelihoodvalid_cert(vector<double> class_, vector<double> valid_cert, vector<vector<double> > count_class) {
    vector<vector<double> > lh_valid_cert(2, vector<double>(2, 0)); 

    for(int i = 0; i < class_.size(); i++) {
		if(class_.at(i) == 0) {
			if(valid_cert.at(i) == 0) {
				lh_valid_cert.at(0).at(0)++;
			}
			else if(valid_cert.at(i) == 1) {
				lh_valid_cert.at(0).at(1)++;
			}
			else {}
		}
		else if(class_.at(i) == 1) {
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
				lh_valid_cert.at(i).at(j) = lh_valid_cert.at(i).at(j) / count_class.at(0).at(0);
			}
			else if(i == 1) {
				lh_valid_cert.at(i).at(j) = lh_valid_cert.at(i).at(j) / count_class.at(0).at(1);
			}
		}
	}
	
	return lh_valid_cert;
}

vector<vector<double> > confusionMatrix(vector<double> matA, vector<double> matB) {

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

// Method that calculate used_days mean 

vector<vector<double> > used_daysMean(vector<double> class_, vector<double> used_days, vector<vector<double> > count_class) {

    vector<vector<double> > mean(1, vector<double> (2, 0));

    for(int i = 0; i < class_.size(); i++) {
        if(class_.at(i) == 0) {
            mean.at(0).at(0) += used_days.at(i);
        }
        else if(class_.at(i) == 1) {
            mean.at(0).at(1) += used_days.at(i);
        }
        else{}
    }


    for(int i = 0; i < mean.size(); i++) {
        for(int j = 0; j < mean[i].size(); j++) {
            if(j == 0) {
                mean.at(i).at(j) = mean.at(i).at(j) / count_class.at(0).at(0);

            }
            else if(j == 1) {
                mean.at(i).at(j) = mean.at(i).at(j) / count_class.at(0).at(1);

            }
        }
    }

    return mean;
}


vector<vector<double> > used_daysVar (vector<double> class_, vector<double> used_days, vector<vector<double> > count_class) {

    vector<vector<double> > var(1, vector<double> (2,0));

    vector<vector<double> > mean = used_daysMean(class_, used_days, count_class);

    for(int i = 0; i < class_.size(); i++) {
        if(class_.at(i) == 0){
            var.at(0).at(0) += pow((used_days.at(i) - mean.at(0).at(0)), 2);
        }
        else if(class_.at(i) == 1) {
            var.at(0).at(0) += pow((used_days.at(i) - mean.at(0).at(1)), 2);
        }
        else {}
    }

    for(int i = 0; i < var.size(); i++) {
        for(int j = 0; j < var[i].size(); j++) {
            if(j == 0) {
                var.at(i).at(j) = var.at(i).at(j) / (count_class.at(0).at(0) - 1 );
            }
            else if(j == 1) {
                var.at(i).at(j) = var.at(i).at(j) / (count_class.at(0).at(1) - 1);
            }
            else {}
        }
    }

    return var;
}

vector<vector<double> > used_days_metrics (vector<vector<double> > used_daysMean, vector<vector<double> > used_daysVar) {

	
	vector<vector<double> > metrics(2, vector<double>(2, 0));  
	
	metrics.at(0).at(0) = used_daysMean.at(0).at(0);
	metrics.at(0).at(1) = sqrt(used_daysVar.at(0).at(0));
	metrics.at(1).at(0) = used_daysMean.at(0).at(1);
	metrics.at(1).at(1) = sqrt(used_daysVar.at(0).at(1));
	
	return metrics;
}




double calc_used_days_lh(double v, double mean_v, double var_v) {
    double used_days_lh = 0;

    used_days_lh = (1 / (sqrt(2 * M_PI * var_v))) * exp( -(pow((v - mean_v), 2)) / (2*var_v));

    return used_days_lh;;
}

vector<vector<double> > calc_raw_prob(double type_doc, double valid_cert, double used_days, vector<vector<double> > apriori, vector<vector<double> > lh_type_doc, vector<vector<double> > lh_valid_cert, vector<vector<double> > used_days_mean,vector<vector<double> > used_days_var) {

    vector<vector<double> > raw(1, vector<double> (2, 0));

    double num_s = lh_type_doc.at(1).at(type_doc-1) * lh_valid_cert.at(1).at(valid_cert) * apriori.at(0).at(1) *
					calc_used_days_lh(used_days, used_days_mean.at(0).at(1), used_days_var.at(0).at(1));		
					
	
	double num_p = lh_type_doc.at(0).at(type_doc-1) * lh_valid_cert.at(0).at(valid_cert) * apriori.at(0).at(0) *
					calc_used_days_lh(used_days, used_days_mean.at(0).at(0), used_days_var.at(0).at(0));				

	double denominator = lh_type_doc.at(1).at(type_doc-1) * lh_valid_cert.at(1).at(valid_cert) *
					calc_used_days_lh(used_days, used_days_mean.at(0).at(1), used_days_var.at(0).at(1)) * apriori.at(0).at(1)
					+ lh_type_doc.at(0).at(type_doc-1) * lh_valid_cert.at(0).at(valid_cert) * 
					calc_used_days_lh(used_days, used_days_mean.at(0).at(0), used_days_var.at(0).at(0)) * apriori.at(0).at(0);

    raw.at(0).at(1) = num_s / denominator;
    raw.at(0).at(0) = num_p / denominator;
}

double accuracy(vector<double> matA, vector<double> matB) {
    int matARow = matA.size();
    int matBRow = matB.size();

    if((matARow != matBRow)) {
        cout << "Error: dimensions need to be the same!" << endl;
    }

    double sum = 0;


    for(int i = 0; i < matA.size(); i++) {
        if(matA.at(i) == matB.at(i)) {
            sum++;
        }
    }

    return sum / matA.size();
}




