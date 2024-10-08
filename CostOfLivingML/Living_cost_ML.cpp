#include <bits/stdc++.h>
using namespace std;

// Class to manipulate data
class LoadCsv {
protected:
    vector<pair<string, vector<float>>> data;
    vector<float> x, y, yp, y_gra, x_test, y_test;

    string file_name;

    float x_factor, y_factor;
    int n;

public:
    void read(string file) {
        file_name = file;
        cout << "CSV Archive: " << file_name << endl;

        fstream fo;

        fo.open(file_name.c_str(), ios::in);

        string line, colname;

        if (fo.good()) {
            getline(fo, line);
            stringstream s(line);

            while (getline(s, colname, ',')) {
                data.push_back({colname, vector<float>{}});
            }

            int temp1;
            while (getline(fo, line)) {
                int col = 0;
                stringstream s1(line);

                while (s1 >> temp1) {
                    data.at(col).second.push_back(temp1);
                    col++;

                    if (s1.peek() == ',') {
                        s1.ignore();
                    }
                }
            }

            for (int i = 0; i < data[0].second.size(); i++) {
                x.push_back(data[0].second.at(i));
            }

            for (int i = 0; i < data[1].second.size(); i++) {
                y.push_back(data[1].second.at(i));
            }
        } else {
            cout << "Error opening file" << endl;
        }

        fo.close();
        
        check();
        data_normalization();
        split_data();
    }

        void check() {
        int i = 0;
        
        while (i < data[0].second.size()) {
            if (x.at(i) != data[0].second.at(i)) {
                cout << "Error in x" << endl;
                break;
            }
            i++;  
        }

        i = 0;  

        while (i < data[1].second.size()) {
            if (y.at(i) != data[1].second.at(i)) {
                cout << "Error in y" << endl;
                break;
            }
            i++;  
        }
        }

        void data_normalization(int normal_lvl = 7, int type = 0) {

            long double x_sum = 0, y_sum = 0;

            for(int i = 0; i < x.size(); i++) {
                x_sum += x[i];
                y_sum += y[i];
            }

            x_factor = x_sum / (normal_lvl * x.size());
            y_factor = y_sum / (normal_lvl * y.size());

            for(int i = 0; i < x.size(); i++) {
                x[i] = x[i] / x_factor;
                y[i] = y[i] / y_factor;
            }
        }

        void split_data(int debug = 0) {
            int total =  0.3 * x.size();
            srand(time(0));

            if(debug) {
                cout << "Indexes " << endl;
            }

        for(int i = 0; i < total; i++) {

            int x_mod = x.size();
            int y_mod = y.size();

            int temp = rand();
            int x_index = temp % x_mod;
            int y_index = temp % y_mod;

            if(debug) {
                cout << x_index << endl;
            }

            x_test.push_back(x[x_index]);
            y_test.push_back(y[y_index]);

            x.erase(x.begin() +  x_index);
            y.erase(y.begin() +  y_index);

        }

        if(debug) {
            cout << "X_TEST" << endl;

        }

        if(debug) {
            for(int i = 0; i < x_test.size(); i++) {
                cout << x_test[i] << endl;
            }
            
        }

        cout << endl;
        }

        void set_size() {
            n = x.size();
        }
    };


// Linear regression class

class LinearRegression:public LoadCsv {
    protected:

        float m, c ,xbar, ybar;
        float slope, intercept;

    public:

        LinearRegression() {
            m = 0;
            c = 0;
            xbar = 0;
            ybar = 0;

        }

        void calculate() {
            float z, q, s = 0, d = 0;
            float siz = y.size();

            xbar = accumulate(x.begin(), x.end(), 0) / siz;
            ybar = accumulate(y.begin(), y.end(), 0) / siz;

            for(int i = 0; i < n; i++) {
                z = (x[i] - xbar);
                q = (y[i] - ybar);
                s = s + (z * q);
                d = d + z * z;
            }

            m = (s / d);
            c = ybar - m * xbar;
        }

        void show() {
            cout << "Slope from Regression Line: " << m << endl;
            cout << "Intercept from Regression Line: " << c << endl;
        }

        void predicted() {
            int i;

            for(i = 0; i < y_test.size(); i++) {
                float z;

                z = m * x_test[i] + c;

                yp.push_back(z);
            }
        }

    void gradient_descent() {
        vector<float> error;
        float b0 = 0;  // Inicial intercept
        float b1 = 0;  // Inicial slope
        float alpha = 0.0019;  // Learning rate
        int max_learning_loops = 4500;

        for (int j = 0; j < max_learning_loops; j++) {
            float err = 0;
            float grad_b0 = 0;
            float grad_b1 = 0;

            for (int i = 0; i < n; i++) {
                float p = b0 + b1 * x[i];  // Predict
                float residual = p - y[i];
                err += residual * residual;  
                grad_b0 += residual;  
                grad_b1 += residual * x[i];  
            }

            b0 -= (alpha / n) * grad_b0;
            b1 -= (alpha / n) * grad_b1;

            error.push_back(err / n);
    }

    cout << "Final values: " << "c (intercept) --> " << b0 << " " << "m (slope) = " << b1 << endl;
    slope = b1;
    intercept = b0;
}


        void predict_gradient(){
        	for(int i = 0;i < x_test.size();i++)
        	{
            	y_gra.push_back((x_test[i] * slope) + intercept);
        	}
    	}
};

// Model evaluate class

class Accuracy:public LinearRegression {

    protected:

        float r2f, r2g, r;

    public:

        Accuracy(){
            r = 0;
            r2f = 0;
            r2g = 0;
        }

    void correlation() {
        float z, q, s = 0, d = 0, siz = n, b = 0, sq;

        for(int i = 0; i < y.size(); i++){ 
            z = (x[i] - xbar);
            q = (y[i] - ybar);
            s = s + (z * q);
            d = d + z * z;
            b += (q * q);
        }

        sq = sqrtf(d * b);

        r = (s / sq);

        cout << "Correlation: " << r << endl;

        if(r > -0.5 && r < -1.0){

            cout << "Strong negative relation." << endl;
            cout << "Negative slope." << endl;
        }

        else if(r >= -0.5 && r <= 0.5) {
            cout << "There's no relation between the variables." << endl;
            cout << "These data should not be used in Linear Regression." << endl;
        }

        else if(r > 0.5 && r <= 1.0) {
            cout << "Strong positive relation" << endl;
            cout << "Positive slope." << endl;
        }
        cout << endl;
    }

    void rsquare_for_formula(){
        float ytbar = accumulate(y_test.begin(), y_test.end(), 0) / y_test.size();

        float z = 0, s = 0;

        for(int i = 0; i < y_test.size(); i++) {
            z += ((yp[i] - y_test[i]) * (yp[i] - y_test[i]));
            s += ((y[i] - ytbar) * (y[i] - ytbar));
        }

        r2f = 1 - (z / s);

        cout << "R2 coefficient (model): " << r2f << endl;
    }

    void rsquare_for_gradient() {

        float numg, deng = 0;
        float y_t_bar = accumulate(y_test.begin(), y_test.end(), 0) / y_test.size();


        for(int i = 0; i < x_test.size(); i++) {
            float temp = (x_test[i] * slope) + intercept;
            numg += ((temp - y_test[i]) * (temp - y_test[i]));
            deng += ((y_test[i] - y_t_bar) * (y_test[i] - y_t_bar));
        }

        cout << "R2 coefficient (gradient): " << 1 - (numg / deng) << endl;
    }

};

int main() {

    Accuracy acc;
    acc.read("data/dataset.csv");
    acc.set_size();
    acc.gradient_descent();
    acc.predict_gradient();
    acc.rsquare_for_gradient();
    acc.correlation();
    acc.calculate();
    acc.show();
    acc.predicted();
    acc.rsquare_for_formula();
}