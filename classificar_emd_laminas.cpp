#include <iostream>
#include <vector>
#include <limits>
#include <fstream>
#include <string>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <stdio.h>      
#include "opencv2/opencv.hpp"
#include <time.h>
#include <bits/stdc++.h> 

clock_t tempo, end;
    
using namespace cv;   
using namespace std;  

cv::Mat srcGD;
cv::Mat src;

// Cores das laminas BGR
int argila[3] = {129,129,129};
int rocha[3]  = {153,0, 153};
int poro[3]   = {255,255,255};

vector<Mat> argilaPatches;
vector<Mat> rochaPatches;
vector<Mat> poroPatches;

int limiar = 48;

struct posLAB {
  int L;
  int a;
  int b;
  int n;
};

/*
* Funcao que realiza a leitura dos patches
**/
void readPatche(){
	ifstream readFile;
	string output;
	string line;
	int dim, size;

	dim = 0;
	size = 0;
	readFile.open("treino/argila.txt");
	if(readFile.is_open()){
		
		readFile >> output;

		dim = atoi(output.c_str());

		for (int ii = 0; ii < dim; ii++){
			
			readFile >> output;
			size = atoi(output.c_str());

			Mat sig(size, 4, CV_32FC1);

			for(int j = 0; j < size; j++){
			
				readFile >> output;
				sig.at< float>( j, 0) = atoi(output.c_str());
				
				readFile >> output;
				sig.at< float>( j, 1) = atoi(output.c_str());
					
				readFile >> output;
				sig.at< float>( j, 2) = atoi(output.c_str());

				readFile >> output;
  				sig.at< float>( j, 3) = atoi(output.c_str());
			}

			argilaPatches.push_back(sig);	
		}
	}


	readFile.close();

	dim = 0;
	size = 0;
	readFile.open("treino/rocha.txt");
	if(readFile.is_open()){
		
		readFile >> output;

		dim = atoi(output.c_str());

		for (int ii = 0; ii < dim; ii++){
				
			readFile >> output;
			size = atoi(output.c_str());

				

			Mat sig(size, 4, CV_32FC1);

			for(int j = 0; j < size; j++){
					
				readFile >> output;
				sig.at< float>( j, 0) = atoi(output.c_str());
					
				readFile >> output;
				sig.at< float>( j, 1) = atoi(output.c_str());
					
				readFile >> output;
				sig.at< float>( j, 2) = atoi(output.c_str());

				readFile >> output;
				sig.at< float>( j, 3) = atoi(output.c_str());
			}

			rochaPatches.push_back(sig);			
		}
	}


	readFile.close();

	dim = 0;
	size = 0;
	readFile.open("treino/poro.txt");
	if(readFile.is_open()){
		
		readFile >> output;

		dim = atoi(output.c_str());

		for (int ii = 0; ii < dim; ii++){
				
			readFile >> output;
			size = atoi(output.c_str());

			Mat sig(size, 4, CV_32FC1);

			for(int j = 0; j < size; j++){
					
				readFile >> output;
				sig.at< float>( j, 0) = atoi(output.c_str());
				
				readFile >> output;
				sig.at< float>( j, 1) = atoi(output.c_str());
					
				readFile >> output;
				sig.at< float>( j, 2) = atoi(output.c_str());

				readFile >> output;
				sig.at< float>( j, 3) = atoi(output.c_str());
			}

			poroPatches.push_back(sig);

			
		}
	}
	readFile.close();
}

/*
* Funcao que verifica se o patch é fundo ou nao.
**/
bool checkColorPatch(Mat patch){

	for (int y = 0; y < patch.rows; y++)
    {
        for (int x = 0; x < patch.cols; x++)
        {

        	if((int) patch.at<Vec3b>(y, x).val[0] > 1 || (int) patch.at<Vec3b>(y, x).val[1] > 1 || (int)patch.at<Vec3b>(y, x).val[2] > 1 ){
        		return true;
        	} 
        }
    }

    return false;
}

/*
* Funcao que verifica se o patch esta adicionado
* caso contrario adiciona o patch na lista de patches
**/
vector<posLAB> checkPosLAB(int l, int a, int b, vector<posLAB> posLabs){
	for(int i = 0; i < posLabs.size(); i++){
		if(a == posLabs[i].a && b == posLabs[i].b && l == posLabs[i].L){
			posLabs[i].n++;
			return posLabs;
		}
	}

	posLAB aux;
	aux.L = l;
	aux.a = a;
	aux.b = b;
	aux.n = 1;
	posLabs.push_back(aux);
	
	return posLabs;
}

/*
* Funcao que converte a estruta Mat em 
* Signal. Para poder usar a EMD.
**/
Mat convertInSignal(Mat patch_){
	vector<posLAB> posLabs;
	for (int y = 0; y < patch_.rows; y++)
    {
        for (int x = 0; x < patch_.cols; x++)
        {
        	posLabs = checkPosLAB(patch_.at<Vec3b>(y, x).val[0], patch_.at<Vec3b>(y, x).val[1], patch_.at<Vec3b>(y, x).val[2], posLabs);
        }
    }

    Mat sig(posLabs.size(), 4, CV_32FC1);
    
    for(int i = 0; i < posLabs.size(); i++){
    	sig.at< float>( i, 0) 	= posLabs[i].n;

       	if ((posLabs[i].L == 0 || posLabs[i].L == 1) && posLabs[i].a == 128 && posLabs[i].b == 128) {
    		for(int j = 0; j < posLabs.size(); j++){
    			if (posLabs[j].L > 1){
    				sig.at< float>( i, 1) = posLabs[j].L;
   					sig.at< float>( i, 2) = posLabs[j].a;
   					sig.at< float>( i, 3) = posLabs[j].b;
   					break;
    			}
    		}
    	}
    	else{
   			sig.at< float>( i, 1) = posLabs[i].L;
   			sig.at< float>( i, 2) = posLabs[i].a;
   			sig.at< float>( i, 3) = posLabs[i].b;
   		}
    }

    return sig;
}

/*
* Funcao responsavel por comprar dois patches
* usando a metrica EMD
**/
float comparePatchEMD(Mat patch1, Mat patch2){
    return cv::EMD(patch1, patch2, CV_DIST_L2); 
}

/*
* Funcao responsavel por extrair os patches 
* de uma imagem. Realizar a classificacao desses patchs
* e calcular os valores das metricas.
*/
void classificationAndCalcs(Mat imgO, Mat imgLAB, int size, string arg){

	Mat imgPatch;
	Mat patchAux;
	cv :: Rect rect;
	int pos = 0;
	int val = INT_MAX;
	int val2;

	Mat imgO2 = imgO.clone();

	for (int y = 0; y + size< imgO.rows; y= y + size){
        for (int x = 0; x + size< imgO.cols; x = x + size){

        	rect =   cv :: Rect ( x , y , size, size);
        	imgPatch = cv :: Mat ( imgO , rect );
        	        	        	
			if(checkColorPatch(imgPatch)){

				//imgPatch = cv :: Mat ( imgLAB , rect );
				Mat patchAux = convertInSignal(cv :: Mat ( imgLAB , rect ));
				val = INT_MAX;
				pos = 10;		
				
				for(int i = 0; i < argilaPatches.size(); i++){
					val2 = comparePatchEMD(patchAux, argilaPatches[i]);
					
					if(val2 < val ){
						val = val2;
						pos = 1;
					}
				}

				for(int i = 0; i < rochaPatches.size(); i++){
					val2 = comparePatchEMD(patchAux, rochaPatches[i]);
					
					if(val2 < val ){
						val = val2;
						pos = 2;
					}
				}

				for(int i = 0; i < poroPatches.size(); i++){
					val2 = comparePatchEMD(patchAux, poroPatches[i]);
					
					if(val2 < val ){
						val = val2;
						pos = 3;
					}
				}

				if(val > limiar) pos = 10;

				for (int yy = y; yy < y + size; yy++){
		        	for (int xx = x; xx< x + size; xx++){

						/*if (imgO.at<Vec3b>(yy, xx).val[0] == 0 && imgO.at<Vec3b>(yy, xx).val[1] == 0 && imgO.at<Vec3b>(yy, xx).val[2] == 0){
							imgO2.at<Vec3b>(yy, xx).val[0] = 0;
							imgO2.at<Vec3b>(yy, xx).val[1] = 0;
							imgO2.at<Vec3b>(yy, xx).val[2] = 0;
						}
						else*/
						if(pos == 1){
							imgO2.at<Vec3b>(yy, xx).val[0] = argila[0];
							imgO2.at<Vec3b>(yy, xx).val[1] = argila[1] ;
							imgO2.at<Vec3b>(yy, xx).val[2] = argila[2]  ;
						} 

						else
						if(pos == 2){
							imgO2.at<Vec3b>(yy, xx).val[0] = rocha[0];
							imgO2.at<Vec3b>(yy, xx).val[1] = rocha[1];
							imgO2.at<Vec3b>(yy, xx).val[2] = rocha[2];
						} 
						else
						if(pos == 3){
							imgO2.at<Vec3b>(yy, xx).val[0] = poro[0];
							imgO2.at<Vec3b>(yy, xx).val[1] = poro[1];
							imgO2.at<Vec3b>(yy, xx).val[2] = poro[2];
						} 
						/*
						else{
							imgO2.at<Vec3b>(yy, xx).val[0] = 100;		
							imgO2.at<Vec3b>(yy, xx).val[1] = 0;
							imgO2.at<Vec3b>(yy, xx).val[2] = 100;
						}*/
					}
				}
			}	
		}

	}
	/*
    for (int y = 0; y < imgO.rows; y++){
        for (int x = 0; x < imgO.cols; x++){
     		
     		if (imgO.at<Vec3b>(y, x).val[0] == 0 && imgO.at<Vec3b>(y, x).val[1] == 0 && imgO.at<Vec3b>(y, x).val[2] == 0){
     			imgO2.at<Vec3b>(y, x).val[0] = 100;
				imgO2.at<Vec3b>(y, x).val[1] = 0;
				imgO2.at<Vec3b>(y, x).val[2] = 100;
			}
        }
    }*/   

 	end = clock();

 	cout << double(end - tempo) / CLOCKS_PER_SEC << endl;

	//cv::imwrite("imagens/_" + arg,imgO2);
}

int main(int argc, char** argv){   
	
    int sizePatch = 11;

	readPatche();
	cout << "Arquivo == " << argv[1] << " e " << argv[2] << endl;
	Mat imgOri = imread(argv[1], 1 )/4; 
	srcGD   = imread(argv[2], 1 );
 	Mat imgLAB ; 
 	cv::cvtColor(imgOri, imgLAB,CV_BGR2Lab);
	tempo = clock();
	classificationAndCalcs(imgOri, imgLAB, sizePatch, argv[2]);
	cout << endl << endl;  
 	return 0;   
}