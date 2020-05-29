#include <iostream>
#include <vector>
#include <limits>
#include <fstream>
#include <string>
#include <stdio.h>      
#include "opencv2/opencv.hpp"
#include <time.h>
#include <stdio.h>
    
using namespace cv;   
using namespace std;   

struct posLAB {
  int L;
  int a;
  int b;
  int n;
};


// Cores das laminas BGR
int argila[3] = {129,129,129};
int rocha[3]  = {153,0, 153};
int poro[3]   = {255,255,255};

/** Modificar o Limiar de Treino **/
int limiar = 5;

vector<Mat> argilaPatches;
vector<Mat> rochaPatches;
vector<Mat> poroPatches;


/*
* Funcao responsavel por salvar a informaao de um patch em um 
* arquivo de texto.
**/
void saveDataPatcheinFile(vector<Mat> patch, string fileName){
	ofstream file;
	file.open(fileName.c_str());
	file << patch.size() << endl;
	for(int i = 0; i < patch.size(); i++){
		file << patch[i].rows << endl;
		for (int k = 0; k < patch[i].rows; k++){
			file << patch[i].at< float>( k, 0) << " " << patch[i].at< float>( k, 1) << " " <<  patch[i].at< float>( k, 2) << " " << patch[i].at< float>( k, 3) << "  ";
		}
		file << endl;
	}
	file.close();
}

/*
* Funcao para salvar os dados de todos os patches
**/
void savePatchesClasses(){		
	saveDataPatcheinFile(argilaPatches, "treinoPatches/argila.txt");	
	saveDataPatcheinFile(rochaPatches, "treinoPatches/rocha.txt");
	saveDataPatcheinFile(poroPatches, "treinoPatches/poro.txt");
}

/*
* Funcao que exibe as informacoes dos patches
**/
void printInfoPatches(){
	cout << "Argila Patches == " << argilaPatches.size() << endl;
	cout << "Rocha Patches == " << rochaPatches.size() << endl;
	cout << "Poro Patches == " << poroPatches.size() << endl;
}

/*
* Funcao que verifica se o patch e representativo
* e retorna a que classe de tecido ele pertence (1..10)
* Se nao for representativo retorna 0.
**/
int checkColorPatch(Mat patch){

	for (int y = 0; y < patch.rows; y++){
        for (int x = 0; x < patch.cols; x++){
        	if(patch.at<Vec3b>(y, x).val[0] != patch.at<Vec3b>(0, 0).val[0] || patch.at<Vec3b>(y, x).val[1] != patch.at<Vec3b>(0, 0).val[1] || patch.at<Vec3b>(y, x).val[2] != patch.at<Vec3b>(0, 0).val[2] ){
        		return 0;
        	} 
        }
    }

    if (patch.at<Vec3b>(0, 0).val[0] == argila[0] && patch.at<Vec3b>(0, 0).val[1] == argila[1] && patch.at<Vec3b>(0, 0).val[2] == argila[2])
    	return 1;
    if (patch.at<Vec3b>(0, 0).val[0] == rocha[0] && patch.at<Vec3b>(0, 0).val[1] == rocha[1] && patch.at<Vec3b>(0, 0).val[2] == rocha[2])
    	return 2;
    if (patch.at<Vec3b>(0, 0).val[0] == poro[0] && patch.at<Vec3b>(0, 0).val[1] == poro[1] && patch.at<Vec3b>(0, 0).val[2] == poro[2])
    	return 3;
  
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

	for (int y = 0; y < patch_.rows; y++){
        for (int x = 0; x < patch_.cols; x++){
        	posLabs = checkPosLAB(patch_.at<Vec3b>(y, x).val[0], patch_.at<Vec3b>(y, x).val[1], patch_.at<Vec3b>(y, x).val[2], posLabs);
        }
    }

    Mat sig(posLabs.size(), 4, CV_32FC1);

    for(int i = 0; i < posLabs.size(); i++){
    	sig.at< float>( i, 0) = posLabs[i].n;
   		sig.at< float>( i, 1) = posLabs[i].L;
   		sig.at< float>( i, 2) = posLabs[i].a;
   		sig.at< float>( i, 3) = posLabs[i].b;
    }

    return sig;
}

/*
* Funcao responsavel por comparar dois patches
* usando a metrica EMD
**/
float comparePatchEMD(Mat patch1, Mat patch2){
    return cv::EMD(patch1, patch2, CV_DIST_L2); 
}

/*
* Compara a distancia de um patch com um 
* conjunto de patches
**/
bool comparePatchWithBasePatchs(Mat patch, vector<Mat> basePatches, float limit){
	if(basePatches.size() == 0) return true;
	for(int i = 0; i < basePatches.size(); i++){
		if(limit > comparePatchEMD(patch, basePatches[i])){
			return false;
		}
	}
	return true;
}


/*
* Funcao responsavel por extrair os patches 
* representativos para servir de base para
* a classificacao.
*/
void splitPatches(Mat imgO, Mat imgGD, int size){

	Mat imgPatch;
	Mat patchAux;
	cv :: Rect rect;
	
	for (int y = 0; y + size< imgO.rows; y++){
        for (int x = 0; x + size< imgGD.cols; x++){

        	rect =   cv :: Rect ( x , y , size, size);
        	imgPatch = cv :: Mat ( imgGD , rect );
        	
			switch(checkColorPatch(imgPatch)){
				case 0 : 	break;
				case 1 : 	patchAux = convertInSignal(cv :: Mat ( imgO , rect ));
							if(comparePatchWithBasePatchs(patchAux, argilaPatches, limiar))
								argilaPatches.push_back(patchAux);
							break;
				case 2 : 	patchAux = convertInSignal(cv :: Mat ( imgO , rect ));
							if(comparePatchWithBasePatchs(patchAux, rochaPatches, limiar))
								rochaPatches.push_back(patchAux);
							break;
				case 3 : 	patchAux = convertInSignal(cv :: Mat ( imgO , rect ));
							if(comparePatchWithBasePatchs(patchAux, poroPatches, limiar))
								poroPatches.push_back(patchAux);
							break;
			}
        }
    }
}


int main(){   
	
    time_t start,end;
	
	vector<String> vecImgOriginal;
	vector<String> vecImgGD;

 	Mat imgOriginal, imgOriginalLAB, imgGD;   
 	
 	vecImgOriginal.push_back("treino/selecionadas/FOTO010.png");
 	vecImgGD.push_back("treino/selecionadas/ground_truth_FOTO010.png");

 	vecImgOriginal.push_back("treino/selecionadas/FOTO031.png");
 	vecImgGD.push_back("treino/selecionadas/ground_truth_2_FOTO031.png");

 	vecImgOriginal.push_back("treino/selecionadas/FOTO032.png");
	vecImgGD.push_back("treino/selecionadas/ground_truth_2_FOTO032.png");

 	vecImgOriginal.push_back("treino/selecionadas/FOTO046.png");
 	vecImgGD.push_back("treino/selecionadas/ground_truth_2_FOTO046.png");

 	vecImgOriginal.push_back("treino/selecionadas/FOTO047.png");
 	vecImgGD.push_back("treino/selecionadas/ground_truth_FOTO047.png");

 	vector<Mat> imgAuxLab;
 	vector<Mat> imgAuxGD;

 	for (int i = 0; i < vecImgGD.size(); i++){
 		cout << " ************** Lendo as Imagens ************ \n";
 		imgAuxLab.push_back(imread(vecImgOriginal[i]));   
 		imgAuxGD.push_back(imread(vecImgGD[i]));   
 		imgAuxLab[i] = imgAuxLab[i]/4;
 		cv::cvtColor(imgAuxLab[i], imgAuxLab[i],CV_BGR2Lab);
 	}

 	time (&start);
 	for (int i = 0; i < vecImgGD.size(); i++){
 		cout << " ************** IMG " << i + 1 << " ************ \n";
 		splitPatches(imgAuxLab[i], imgAuxGD[i], 3);
 		printInfoPatches(); 		
 	}
 	time (&end);
 	savePatchesClasses();
 	double dif = difftime (end,start);
	printf ("Elasped time is %.2lf seconds.", dif );
 	return 0;   
}
