using namespace std;
#include <iostream>
#include <fstream>
#include <cstring> // for std::strlen
#include <cstddef> // for std::size_t -> is a typedef on an unsinged int
#include<vector>
#include<cmath>
#include<cstdlib>
#include <unistd.h>
#include<bits/stdc++.h>

/*void create_image(CvSize size, int channels, unsigned char data[28][28], int imagenumber) {
string imgname; ostringstream imgstrm;string fullpath;
imgstrm << imagenumber;
imgname=imgstrm.str();
fullpath="D:\\MNIST\\"+imgname+".jpg";

IplImage *imghead=cvCreateImageHeader(size, IPL_DEPTH_8U, channels);
cvSetData(imghead, data, size.width);
cvSaveImage(fullpath.c_str(),imghead);
}*/

unsigned long mix(unsigned long a, unsigned long b, unsigned long c)
{
    a=a-b;  a=a-c;  a=a^(c >> 13);
    b=b-c;  b=b-a;  b=b^(a << 8);
    c=c-a;  c=c-b;  c=c^(b >> 13);
    a=a-b;  a=a-c;  a=a^(c >> 12);
    b=b-c;  b=b-a;  b=b^(a << 16);
    c=c-a;  c=c-b;  c=c^(b >> 5);
    a=a-b;  a=a-c;  a=a^(c >> 3);
    b=b-c;  b=b-a;  b=b^(a << 10);
    c=c-a;  c=c-b;  c=c^(b >> 15);
    return c;
}

int reverseInt (int i)
{
    unsigned char c1, c2, c3, c4;

    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;

    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

int K, n, m;

vector<vector<double> > u;

vector<vector<vector <double> > > weights;

int dataSize;
int totalPicked;
int picked[10];

vector<double> x;
vector<double> y;
vector<double> bias;
vector<double> teaching;

vector<double> x2;
vector<vector<double> > y2;
vector<double> bias2;
vector<double> teaching2;

vector<double> t;

const double e = 2.7182818284590;
const double epsilon = 0.1;
const double bound = 1.0/28.0;

double sygmoidalCinection(double x) {
    return 1/(1 + exp(-x));
}

double derivativeSygmoidalCinection(double x) {
    return sygmoidalCinection(x) * (1 - sygmoidalCinection(x));
}

void initializeWeights(int beforeLayer) {
    if(beforeLayer == 2) {
        weights[0] = vector<vector <double> > (K);
        for(int i = 0; i < K; i++)
            weights[0][i] = vector<double> (n);
    }
    else if(beforeLayer == 3) {
        weights[1] = vector<vector <double> > (n);
        for(int i = 0; i < n; i++)
            weights[1][i] = vector<double> (m);
    }
}

double w(int beforeLayer, int from, int to) {
    return weights[beforeLayer - 2][from][to];
}

void setW(int beforeLayer, int from, int to, double value) {
    weights[beforeLayer - 2][from][to] = value;
}

void mlpBp(int current) {
    //Forward Phase
    for(int j = 0; j < n; j++) {
        x[j] = bias[j];
        for(int i = 0; i < K; i++) {
            x[j] += u[current][i] * w(2, i, j);
        }
        y[j] = sygmoidalCinection(x[j]);
        //y[j] = tanh(x[j]);
    }

    for(int k = 0; k < m; k++) {
        x2[k] = bias2[k];
        for(int j = 0; j < n; j++) {
            x2[k] += y[j] * w(3, j, k);
        }
        y2[current][k] = sygmoidalCinection(x2[k]);
        //y2[current][k] = tanh(x2[k]);
    }

    //Backward Phase
    for(int k = 0; k < m; k++) {
		if (t[current] == k) 
        teaching2[k] = 1 - y2[current][k];
		else
		teaching2[k] = 0 - y2[current][k];
        for(int j = 0; j < n; j++)
            setW(3, j, k, w(3,j,k) + epsilon * y[j] * teaching2[k] * derivativeSygmoidalCinection(x2[k]) );
            //setW(3, j, k, w(3,j,k) + epsilon * y[j] * teaching2[k] * (1 - tanh(x2[k]) * tanh(x2[k])) );
        bias2[k] = bias2[k] + epsilon * teaching2[k] * derivativeSygmoidalCinection(x2[k]);
        //bias2[k] = bias2[k] + epsilon * teaching2[k] * (1 - tanh(x2[k]) * tanh(x2[k]));

    }

    for(int j = 0; j < n; j++) {
        int sum = 0;
        for(int k = 0; k < m; k++) sum += w(3, j, k) * teaching2[k];
        teaching[j] = sum;

        for(int i = 0; i < K; i++)
            setW(2, i, j, w(2, i, j) + epsilon * u[current][i] * teaching[j] * derivativeSygmoidalCinection(x[j]) );
            //setW(2, i, j, w(2, i, j) + epsilon * u[current][i] * teaching[j] * (1 - tanh(x[j]) * tanh(x[j])) );
        bias[j] = bias[j] + epsilon * teaching[j] * derivativeSygmoidalCinection(x[j]);
        //bias[j] = bias[j] + epsilon * teaching[j] * (1 - tanh(x[j]) * tanh(x[j]));
    }
}

void read_mnist(char* filename, char* filename2)
{
    ifstream file; 
	file.open(filename, ios::binary);
    ifstream file2(filename2, ios::binary);
    /*size_t size;
    file.seekg(0, ios::end); // set the pointer to the end
	size = file.tellg() ; // get the length of the file
	cout << "Size of file: " << size << endl;
	file.seekg(0, ios::beg); // set the pointer to the beginning*/
    if (file.is_open())
    {
        int magic_number=0;
        int number_of_images=0;
        int n_rows=0;
        int n_cols=0;

        file.read((char*)&magic_number,sizeof(magic_number));
        magic_number= reverseInt(magic_number);
        if(magic_number != 2051) throw runtime_error("Invalid MNIST image file!");

        file.read((char*)&number_of_images,sizeof(number_of_images));
        number_of_images= reverseInt(number_of_images);

        file.read((char*)&n_rows,sizeof(n_rows));
        n_rows= reverseInt(n_rows);

        file.read((char*)&n_cols,sizeof(n_cols));
        n_cols= reverseInt(n_cols);
		
		cout << "Parameters file: "<< number_of_images << ", " << n_rows << ", " << n_cols << endl;

        int magic_number2 = 0;
        int number_of_labels = 0;
        file2.read((char*)&magic_number2,sizeof(magic_number2));
        magic_number2 = reverseInt(magic_number2);
        if(magic_number2 != 2049) throw runtime_error("Invalid MNIST label file!");

        file2.read((char*)&number_of_labels,sizeof(number_of_labels));
        number_of_labels = reverseInt(number_of_labels);
		
		cout << "Parameters file2: " << number_of_labels << endl;

        for(int i=0;i<number_of_images;++i)
        {
            unsigned char temp2 = 0;
            file2.read((char*)&temp2, sizeof(temp2));
            int label = (int) temp2;

            if(picked[label] == dataSize/10) {
                unsigned char temp3[K]; file.read((char*)&temp3, K);
                continue;
            }
            picked[label]++;
            u[totalPicked] = vector<double> (K);
            t[totalPicked] = label;



            for(int r=0;r<n_rows;++r)
            {
                for(int c=0;c<n_cols;++c)
                {
                    unsigned char temp = 0;
                    //do {
                        file.read((char *) &temp, sizeof(temp));
                    //} while(file.gcount() == 0);
                    int pixel = r*28+c;
                    u[totalPicked][pixel] = (int)temp/255.0;
                    //if(i == 0) cout << (int) temp << "  " << (int)temp/255.0 << endl;
                }
            }
            totalPicked++;
            if(totalPicked == dataSize) break;
        }

    }
    cout << "Done" << endl;
    file.close();
    file2.close();
}

const int range_from  = 0;
const int range_to    = 2000000000;
random_device                  rand_dev;
mt19937                        generator(rand_dev());
uniform_int_distribution<int>  distr(range_from, range_to);

int main() {
    unsigned long seed = mix(clock(), time(NULL), getpid());
    srand(seed);
    /*for(int  i = 0; i < 10; i++) {
        cout << distr(generator) << endl;
    }*/

    int numberOfLayers = 3;
    K = 784; n = 300; m = 10;


    dataSize = 10;
    totalPicked = 0;
    for(int i = 0; i < 10; i++) picked[i] = 0;
    u = vector<vector<double> > (dataSize);
    t = vector<double> (dataSize);
    read_mnist("train-images.idx3-ubyte", "train-labels.idx1-ubyte");
    cout << "Done2" << endl;
    /*
    for(int i = 0; i < dataSize; i++) {
        u[i] = vector<double> (K);
        int ans = 0;
        for(int j = 0; j < K; j++) {
            u[i][j] = ((i & (1<<j)) != 0);
            ans ^= (int) u[i][j];
        }
        if(ans == 1)
            t[i] = ans;
        else
            t[i] = -1;
        //t[i]  = u[i][K-1];
        //if(t[i] == 0) t[i] = -1;
    }
    */

    x = vector<double>(n);
    y = vector<double>(n);
    bias = vector<double>(n);
    teaching = vector<double>(n);

    x2 = vector<double>(m);
    y2 = vector<vector<double> > (dataSize);
    for(int i = 0; i < dataSize; i++)
        y2[i] = vector<double> (m);
    bias2 = vector<double>(m);
    teaching2 = vector<double>(m);

    weights = vector<vector<vector <double> > > (numberOfLayers - 1);
    initializeWeights(2);
    initializeWeights(3);

    /*for(int i = 0; i < n; i++)
        bias[i] = (static_cast<double>(rand()) / RAND_MAX * 2000000000 - 1000000000) / 1000000000.0;
    for(int j = 0; j < m; j++)
        bias2[j] = (static_cast<double>(rand()) / RAND_MAX * 2000000000 - 1000000000) / 1000000000.0;
    for(int i = 0; i < K; i++)
        for(int j = 0; j < n; j++)
            weights[0][i][j] = (static_cast<double>(rand()) / RAND_MAX * 2000000000 - 1000000000) / 1000000000.0;
    for(int i = 0; i < n; i++)
        for(int j = 0; j < m; j++)
            weights[1][i][j] = (static_cast<double>(rand()) / RAND_MAX * 2000000000 - 1000000000) / 1000000000.0;
            */

    for(int i = 0; i < n; i++) bias[i] = (distr(generator) - 1000000000) / 1000000000.0 * bound;
    for(int j = 0; j < m; j++) bias2[j] = (distr(generator) - 1000000000) / 1000000000.0 * bound;
    for(int i = 0; i < K; i++)
        for(int j = 0; j < n; j++)
            weights[0][i][j] = (distr(generator) - 1000000000) / 1000000000.0 * bound;
    for(int i = 0; i < n; i++)
        for(int j = 0; j < m; j++)
            weights[1][i][j] = (distr(generator) - 1000000000) / 1000000000.0 * bound;



        for(int run = 0; run < 1000; run++) {
            for(int test = 0; test < dataSize; test++) {
                mlpBp(test);
            }
            /*cout << "run: " << run+1 << '\n';
                    for(int i = 0; i < (1<<K); i++) {
            for(int j = K-1; j >= 0; j--)
                cout << u[i][j];
            cout << ' ' << t[i] << ' ';
            cout << y2[i][0] << endl;*/
        }
            for(int i = 0; i < dataSize; i++) {
            //for(int j = K-1; j >= 0; j--)
                //cout << u[i][j];
			cout << i+1;
            cout << ' ' << t[i] << ' ';
			cout << y2[i][0];
			for(int j = 1; j < 10; j++)
            cout << ' ' << y2[i][j];
			cout << endl;
        }

    return 0;
}
