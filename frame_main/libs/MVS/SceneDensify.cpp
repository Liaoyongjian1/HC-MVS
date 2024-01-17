/*
* SceneDensify.cpp
*
* Copyright (c) 2014-2015 SEACAVE
*
* Author(s):
*
*      cDc <cdc.seacave@gmail.com>
*
*
* This program is free software: you can redistribute it and/or modify
* it under the terms of the GNU Affero General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU Affero General Public License for more details.
*
* You should have received a copy of the GNU Affero General Public License
* along with this program.  If not, see <http://www.gnu.org/licenses/>.
*
*
* Additional Terms:
*
*      You are required to preserve legal notices and author attributions in
*      that material or in the Appropriate Legal Notices displayed by works
*      containing it.
*/

#include "Common.h"
#include "Scene.h"
#include "SceneDensify.h"
// MRF: view selection
#include "../Math/TRWS/MRFEnergy.h"
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Simple_cartesian.h>
#include <CGAL/linear_least_squares_fitting_3.h>

#include <CGAL/remove_outliers.h>
#include <CGAL/compute_average_spacing.h>
#include <CGAL/Shape_detection/Efficient_RANSAC.h>
#include <CGAL/ch_graham_andrew.h>
#include <CGAL/Polygon_2_algorithms.h>
#include <CGAL/pca_estimate_normals.h>

#include <CGAL/Classification.h>
#include <CGAL/Classification/Cluster.h>
#include <CGAL/bounding_box.h>

#include <CGAL/Polyhedron_3.h>
#include <CGAL/convex_hull_3.h>
#include <CGAL/Point_set_3.h>

//超像素：
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/opencv.hpp>
#include "Lsc.hpp"
#include <ctype.h>
#include <stdio.h>
#include <iostream>

#include <fstream>
#include <cmath>

#include <opencv2/imgproc/types_c.h>

using namespace MVS;
namespace Classification = CGAL::Classification;


// D E F I N E S ///////////////////////////////////////////////////

// uncomment to enable multi-threading based on OpenMP
#ifdef _USE_OPENMP
#define DENSE_USE_OPENMP
#endif


// S T R U C T S ///////////////////////////////////////////////////

// Dense3D data.events
enum EVENT_TYPE {
	EVT_FAIL = 0,
	EVT_CLOSE,

	EVT_PROCESSIMAGE,

	EVT_ESTIMATEDEPTHMAP,
	EVT_OPTIMIZEDEPTHMAP,
	EVT_SAVEDEPTHMAP,

	EVT_FILTERDEPTHMAP,
	EVT_ADJUSTDEPTHMAP,
};

class EVTFail : public Event
{
public:
	EVTFail() : Event(EVT_FAIL) {}
};
class EVTClose : public Event
{
public:
	EVTClose() : Event(EVT_CLOSE) {}
};

class EVTProcessImage : public Event
{
public:
	IIndex idxImage;
	EVTProcessImage(IIndex _idxImage) : Event(EVT_PROCESSIMAGE), idxImage(_idxImage) {}
};

class EVTEstimateDepthMap : public Event
{
public:
	IIndex idxImage;
	EVTEstimateDepthMap(IIndex _idxImage) : Event(EVT_ESTIMATEDEPTHMAP), idxImage(_idxImage) {}
};
class EVTOptimizeDepthMap : public Event
{
public:
	IIndex idxImage;
	EVTOptimizeDepthMap(IIndex _idxImage) : Event(EVT_OPTIMIZEDEPTHMAP), idxImage(_idxImage) {}
};
class EVTSaveDepthMap : public Event
{
public:
	IIndex idxImage;
	EVTSaveDepthMap(IIndex _idxImage) : Event(EVT_SAVEDEPTHMAP), idxImage(_idxImage) {}
};

class EVTFilterDepthMap : public Event
{
public:
	IIndex idxImage;
	EVTFilterDepthMap(IIndex _idxImage) : Event(EVT_FILTERDEPTHMAP), idxImage(_idxImage) {}
};
class EVTAdjustDepthMap : public Event
{
public:
	IIndex idxImage;
	EVTAdjustDepthMap(IIndex _idxImage) : Event(EVT_ADJUSTDEPTHMAP), idxImage(_idxImage) {}
};
/*----------------------------------------------------------------*/


// convert the ZNCC score to a weight used to average the fused points
inline float Conf2Weight(float conf, Depth depth) {
	return 1.f/(MAXF(1.f-conf,0.03f)*depth*depth);
}
/*----------------------------------------------------------------*/


// S T R U C T S ///////////////////////////////////////////////////


DepthMapsData::DepthMapsData(Scene& _scene)
	:
	scene(_scene),
	arrDepthData(_scene.images.GetSize())
{
} // constructor

DepthMapsData::~DepthMapsData()
{
} // destructor
/*----------------------------------------------------------------*/


// globally choose the best target view for each image,
// trying in the same time the selected image pairs to cover the whole scene;
// the map of selected neighbors for each image is returned in neighborsMap.
// For each view a list of neighbor views ordered by number of shared sparse points and overlapped image area is given.
// Next a graph is formed such that the vertices are the views and two vertices are connected by an edge if the two views have each other as neighbors.
// For each vertex, a list of possible labels is created using the list of neighbor views and scored accordingly (the score is normalized by the average score).
// For each existing edge, the score is defined such that pairing the same two views for any two vertices is discouraged (a constant high penalty is applied for such edges).
// This primal-dual defined problem, even if NP hard, can be solved by a Belief Propagation like algorithm, obtaining in general a solution close enough to optimality.
bool DepthMapsData::SelectViews(IIndexArr& images, IIndexArr& imagesMap, IIndexArr& neighborsMap)
{
	// find all pair of images valid for dense reconstruction
	typedef std::unordered_map<uint64_t,float> PairAreaMap;
	PairAreaMap edges;
	double totScore(0);
	unsigned numScores(0);
	FOREACH(i, images) {
		const IIndex idx(images[i]);
		ASSERT(imagesMap[idx] != NO_ID);
		const ViewScoreArr& neighbors(arrDepthData[idx].neighbors);
		ASSERT(neighbors.GetSize() <= OPTDENSE::nMaxViews);
		// register edges
		FOREACHPTR(pNeighbor, neighbors) {
			const IIndex idx2(pNeighbor->idx.ID);
			ASSERT(imagesMap[idx2] != NO_ID);
			edges[MakePairIdx(idx,idx2)] = pNeighbor->idx.area;
			totScore += pNeighbor->score;
			++numScores;
		}
	}
	if (edges.empty())
		return false;
	const float avgScore((float)(totScore/(double)numScores));

	// run global optimization
	const float fPairwiseMul = OPTDENSE::fPairwiseMul; // default 0.3
	const float fEmptyUnaryMult = 6.f;
	const float fEmptyPairwise = 8.f*OPTDENSE::fPairwiseMul;
	const float fSamePairwise = 24.f*OPTDENSE::fPairwiseMul;
	const IIndex _num_labels = OPTDENSE::nMaxViews+1; // N neighbors and an empty state
	const IIndex _num_nodes = images.GetSize();
	typedef MRFEnergy<TypeGeneral> MRFEnergyType;
	CAutoPtr<MRFEnergyType> energy(new MRFEnergyType(TypeGeneral::GlobalSize()));
	CAutoPtrArr<MRFEnergyType::NodeId> nodes(new MRFEnergyType::NodeId[_num_nodes]);
	typedef SEACAVE::cList<TypeGeneral::REAL, const TypeGeneral::REAL&, 0> EnergyCostArr;
	// unary costs: inverse proportional to the image pair score
	EnergyCostArr arrUnary(_num_labels);
	for (IIndex n=0; n<_num_nodes; ++n) {
		const ViewScoreArr& neighbors(arrDepthData[images[n]].neighbors);
		FOREACH(k, neighbors)
			arrUnary[k] = avgScore/neighbors[k].score; // use average score to normalize the values (not to depend so much on the number of features in the scene)
		arrUnary[neighbors.GetSize()] = fEmptyUnaryMult*(neighbors.IsEmpty()?avgScore*0.01f:arrUnary[neighbors.GetSize()-1]);
		nodes[n] = energy->AddNode(TypeGeneral::LocalSize(neighbors.GetSize()+1), TypeGeneral::NodeData(arrUnary.Begin()));
	}
	// pairwise costs: as ratios between the area to be covered and the area actually covered
	EnergyCostArr arrPairwise(_num_labels*_num_labels);
	for (PairAreaMap::const_reference edge: edges) {
		const PairIdx pair(edge.first);
		const float area(edge.second);
		const ViewScoreArr& neighborsI(arrDepthData[pair.i].neighbors);
		const ViewScoreArr& neighborsJ(arrDepthData[pair.j].neighbors);
		arrPairwise.Empty();
		FOREACHPTR(pNj, neighborsJ) {
			const IIndex i(pNj->idx.ID);
			const float areaJ(area/pNj->idx.area);
			FOREACHPTR(pNi, neighborsI) {
				const IIndex j(pNi->idx.ID);
				const float areaI(area/pNi->idx.area);
				arrPairwise.Insert(pair.i == i && pair.j == j ? fSamePairwise : fPairwiseMul*(areaI+areaJ));
			}
			arrPairwise.Insert(fEmptyPairwise+fPairwiseMul*areaJ);
		}
		for (const ViewScore& Ni: neighborsI) {
			const float areaI(area/Ni.idx.area);
			arrPairwise.Insert(fPairwiseMul*areaI+fEmptyPairwise);
		}
		arrPairwise.Insert(fEmptyPairwise*2);
		const IIndex nodeI(imagesMap[pair.i]);
		const IIndex nodeJ(imagesMap[pair.j]);
		energy->AddEdge(nodes[nodeI], nodes[nodeJ], TypeGeneral::EdgeData(TypeGeneral::GENERAL, arrPairwise.Begin()));
	}

	// minimize energy
	MRFEnergyType::Options options;
	options.m_eps = OPTDENSE::fOptimizerEps;
	options.m_iterMax = OPTDENSE::nOptimizerMaxIters;
	#ifndef _RELEASE
	options.m_printIter = 1;
	options.m_printMinIter = 1;
	#endif
	#if 1
	TypeGeneral::REAL energyVal, lowerBound;
	energy->Minimize_TRW_S(options, lowerBound, energyVal);
	#else
	TypeGeneral::REAL energyVal;
	energy->Minimize_BP(options, energyVal);
	#endif

	// extract optimized depth map
	neighborsMap.Resize(_num_nodes);
	for (IIndex n=0; n<_num_nodes; ++n) {
		const ViewScoreArr& neighbors(arrDepthData[images[n]].neighbors);
		IIndex& idxNeighbor = neighborsMap[n];
		const IIndex label((IIndex)energy->GetSolution(nodes[n]));
		ASSERT(label <= neighbors.GetSize());
		if (label == neighbors.GetSize()) {
			idxNeighbor = NO_ID; // empty
		} else {
			idxNeighbor = label;
			DEBUG_ULTIMATE("\treference image %3u paired with target image %3u (idx %2u)", images[n], neighbors[label].idx.ID, label);
		}
	}

	// remove all images with no valid neighbors
	RFOREACH(i, neighborsMap) {
		if (neighborsMap[i] == NO_ID) {
			// remove image with no neighbors
			for (IIndex& imageMap: imagesMap)
				if (imageMap != NO_ID && imageMap > i)
					--imageMap;
			imagesMap[images[i]] = NO_ID;
			images.RemoveAtMove(i);
			neighborsMap.RemoveAtMove(i);
		}
	}
	return !images.IsEmpty();
} // SelectViews
/*----------------------------------------------------------------*/

// compute visibility for the reference image (the first image in "images")
// and select the best views for reconstructing the depth-map;
// extract also all 3D points seen by the reference image
bool DepthMapsData::SelectViews(DepthData& depthData)
{
	// find and sort valid neighbor views
	const IIndex idxImage((IIndex)(&depthData-arrDepthData.Begin()));
	ASSERT(depthData.neighbors.IsEmpty());
	ASSERT(scene.images[idxImage].neighbors.IsEmpty());
	if (!scene.SelectNeighborViews(idxImage, depthData.points, OPTDENSE::nMinViews, OPTDENSE::nMinViewsTrustPoint>1?OPTDENSE::nMinViewsTrustPoint:2, FD2R(OPTDENSE::fOptimAngle)))
		return false;
	depthData.neighbors.CopyOf(scene.images[idxImage].neighbors);

	// remove invalid neighbor views
	const float fMinArea(OPTDENSE::fMinArea);
	const float fMinScale(0.2f), fMaxScale(3.2f);
	const float fMinAngle(FD2R(OPTDENSE::fMinAngle));
	const float fMaxAngle(FD2R(OPTDENSE::fMaxAngle));
	if (!Scene::FilterNeighborViews(depthData.neighbors, fMinArea, fMinScale, fMaxScale, fMinAngle, fMaxAngle, OPTDENSE::nMaxViews)) {
		DEBUG_EXTRA("error: reference image %3u has no good images in view", idxImage);
		return false;
	}
	return true;
} // SelectViews
/*----------------------------------------------------------------*/

// select target image for the reference image (the first image in "images")
// and initialize images data;
// if idxNeighbor is not NO_ID, only the reference image and the given neighbor are initialized;
// if numNeighbors is not 0, only the first numNeighbors neighbors are initialized;
// otherwise all are initialized;
// returns false if there are no good neighbors to estimate the depth-map
bool DepthMapsData::InitViews(DepthData& depthData, IIndex idxNeighbor, IIndex numNeighbors)
{
	const IIndex idxImage((IIndex)(&depthData-arrDepthData.Begin()));
	ASSERT(!depthData.neighbors.IsEmpty());
	ASSERT(depthData.images.IsEmpty());
	


	// set this image the first image in the array
	depthData.images.Reserve(depthData.neighbors.GetSize()+1);
	depthData.images.AddEmpty();


	if (idxNeighbor != NO_ID) {
		// set target image as the given neighbor
		const ViewScore& neighbor = depthData.neighbors[idxNeighbor];
		DepthData::ViewData& imageTrg = depthData.images.AddEmpty();
		imageTrg.pImageData = &scene.images[neighbor.idx.ID];
		imageTrg.scale = neighbor.idx.scale;
		imageTrg.camera = imageTrg.pImageData->camera;
		imageTrg.pImageData->image.toGray(imageTrg.image, cv::COLOR_BGR2GRAY, true);
		if (imageTrg.ScaleImage(imageTrg.image, imageTrg.image, imageTrg.scale))
			imageTrg.camera = imageTrg.pImageData->GetCamera(scene.platforms, imageTrg.image.size());
		DEBUG_EXTRA("Reference image %3u paired with image %3u", idxImage, neighbor.idx.ID);
	} else {
		// init all neighbor views too (global reconstruction is used)
		const float fMinScore(MAXF(depthData.neighbors.First().score*(OPTDENSE::fViewMinScoreRatio*0.1f), OPTDENSE::fViewMinScore));
		FOREACH(idx, depthData.neighbors) {
			const ViewScore& neighbor = depthData.neighbors[idx];
			if ((numNeighbors && depthData.images.GetSize() > numNeighbors) ||
				(neighbor.score < fMinScore))
				break;
			DepthData::ViewData& imageTrg = depthData.images.AddEmpty();
			imageTrg.pImageData = &scene.images[neighbor.idx.ID];
			imageTrg.scale = neighbor.idx.scale;
			imageTrg.camera = imageTrg.pImageData->camera;
			imageTrg.pImageData->image.toGray(imageTrg.image, cv::COLOR_BGR2GRAY, true);
			if (imageTrg.ScaleImage(imageTrg.image, imageTrg.image, imageTrg.scale))
				imageTrg.camera = imageTrg.pImageData->GetCamera(scene.platforms, imageTrg.image.size());
		}
		#if TD_VERBOSE != TD_VERBOSE_OFF
		// print selected views
		if (g_nVerbosityLevel > 2) {
			String msg;
			for (IIndex i=1; i<depthData.images.GetSize(); ++i)
				msg += String::FormatString(" %3u(%.2fscl)", depthData.images[i].GetID(), depthData.images[i].scale);
			VERBOSE("Reference image %3u paired with %u views:%s (%u shared points)", idxImage, depthData.images.GetSize()-1, msg.c_str(), depthData.points.GetSize());
		} else
		DEBUG_EXTRA("Reference image %3u paired with %u views", idxImage, depthData.images.GetSize()-1);
		#endif
	}
	if (depthData.images.GetSize() < 2) {
		depthData.images.Release();
		return false;
	}

	// init the first image as well
	DepthData::ViewData& imageRef = depthData.images.First();
	imageRef.scale = 1;
	imageRef.pImageData = &scene.images[idxImage];
	imageRef.pImageData->image.toGray(imageRef.image, cv::COLOR_BGR2GRAY, true);
	imageRef.camera = imageRef.pImageData->camera;
	// cv::imwrite("imageRef.png",imageRef.pImageData->image);
	// cvWaitKey(2000);
	


	
	std::vector<cv::Mat> imgs,grayImgs;
	depthData.flow_images.Reserve(depthData.images.GetSize()-1);
	imgs.push_back(imageRef.pImageData->image);
	// cv::imwrite("imageRef.png",imageRef.pImageData->image);

	// FOREACH(idx, depthData.images){
	// 	std::string name = cv::format("r%d.png",idx);
	// 	cv::imwrite(name,depthData.images[idx].pImageData->image);
	// }

	// img = imread("/home/l/data/pipes/images/dslr_images_undistorted/DSC_0635.JPG");
	// FOREACH(idx, depthData.images){
	// 	if (idx==0)
	// 	continue;
		
		depthData.flow_images.AddEmpty();
		imgs.push_back(depthData.images[1].pImageData->image);
		cv::resize(imgs[1] , imgs[1] , imgs[0].size() , 0, 0, cv::INTER_NEAREST);
		// imgs.push_back(depthData.images[idx].pImageData->image);
		// std::string name1 = cv::format("neig%d.png",idx);
		// cv::imwrite(name1,depthData.images[idx].pImageData->image);
		// cvWaitKey(2000);
		
		for(size_t i=0;i<imgs.size();i++){
			
			cv::Mat temp;
			temp.create(imgs[i].rows, imgs[i].cols, CV_8UC1);

			cvtColor(imgs[i], temp, CV_RGB2GRAY);
			grayImgs.push_back(temp);
		}
		
		//for(size_t i=0;i<imgs.size()&&i<grayImgs.size();i++){
			// imwrite("origin.png",imgs[i]);
			// imwrite("gray.png",grayImgs[i]);
		//}

		
		// std::vector<Point2f> point[2];
		// double qualityLevel = 0.01;
		// double minDistance = 10;
		/*
			void goodFeaturesToTrack( InputArray image, OutputArray corners,
									int maxCorners, double qualityLevel, double minDistance,
									InputArray mask=noArray(), int blockSize=3,
									bool useHarrisDetector=false, double k=0.04 )
		*/
		
		// goodFeaturesToTrack(grayImgs[0], point[0], 1000, qualityLevel, minDistance);
		// std::cout<<point[0].size()<<std::endl;
		/*
		void circle(CV_IN_OUT Mat& img, Point center, int radius, const Scalar& color, int thickness=1, int lineType=8, int shift=0);
		*/
		
		//for(size_t i= 0;i<point[0].size();i++){
		// circle(imgs[0], cvPoint(cvRound(point[0][i].x),cvRound(point[0][i].y)), 3, cvScalar(255, 0, 0), 1, CV_AA, 0);
		//}
		//imshow("detected corner", imgs[0]);
		/*
		void cv::calcOpticalFlowFarneback( InputArray _prev0, InputArray _next0,
								OutputArray _flow0, double pyr_scale, int levels, int winsize,
								int iterations, int poly_n, double poly_sigma, int flags )
		void line(Mat& img, Point pt1, Point pt2, const Scalar& color, int thickness=1, int lineType=8, int shift=0)
		*/
		
		cv::Mat flow;
		calcOpticalFlowFarneback(grayImgs[0], grayImgs[1], flow, 0.5, 3, 15, 3, 5, 1.2, 0);
		std::cout<<std::endl;
		std::cout<<flow.size()<<std::endl;  
		
		for(size_t y=0;y<imgs[0].rows;y+=1){
			for(size_t x=0;x<imgs[0].cols;x+=1){
				Point2f fxy = flow.at<Point2f>(y, x);
				depthData.flow_images[0].x0.x = fxy.x;
				depthData.flow_images[0].x0.y = fxy.y;
				// depthData.flow_images[idx-1].x0.x = fxy.x;
				// depthData.flow_images[idx-1].x0.y = fxy.y;
				// line(imgs[0], cv::Point(x,y), cv::Point(cvRound(x+fxy.x), cvRound(y+fxy.y)), CV_RGB(0, 255, 0), 1, 8);
			}
		}

		// for(size_t y=0;y<imgs[0].rows;y+=5){
		// 	for(size_t x=0;x<imgs[0].cols;x+=5){
		// 		Point2f fxy = flow.at<Point2f>(y, x);
		// 		// depthData.flow_images[idx-1].x0.x = fxy.x;
		// 		// depthData.flow_images[idx-1].x0.y = fxy.y;
		// 		line(imgs[0], cv::Point(x,y), cv::Point(cvRound(x+fxy.x), cvRound(y+fxy.y)), CV_RGB(0, 255, 0), 1, 8);
		// 	}
		// }
		// std::string name = cv::format("n%d.png",idx);
		// imwrite(name, imgs[0]);
		
		// TermCriteria criteria = TermCriteria(TermCriteria::COUNT|TermCriteria::EPS, 20, 0.03);
		// vector<uchar> status;
		// vector<float> err;

		// calcOpticalFlowPyrLK(grayImgs[0], grayImgs[1], point[0], point[1], status, err, Size(15, 15), 3, criteria);

		// for(size_t i=0;i<point[0].size()&&i<point[1].size();i++){
		//     line(imgs[1],Point(cvRound(point[0][i].x),cvRound(point[0][i].y)), Point(cvRound(point[1][i].x),
		//         cvRound(point[1][i].y)), cvScalar(0,50,200),1,CV_AA);
		// }
	// }
	
	return true;
} // InitViews
/*----------------------------------------------------------------*/

// roughly estimate depth and normal maps by triangulating the sparse point cloud
// and interpolating normal and depth for all pixels
bool DepthMapsData::InitDepthMap(DepthData& depthData)
{
	TD_TIMER_STARTD();

	ASSERT(depthData.images.GetSize() > 1 && !depthData.points.IsEmpty());
	const DepthData::ViewData& image(depthData.GetView());


	if(OPTDENSE::initTriangulate){
		TriangulatePoints2DepthMap(image, scene.pointcloud, depthData.points, depthData.depthMap, depthData.normalMap, depthData.dMin, depthData.dMax);
		depthData.dMin *= 0.9f;
		depthData.dMax *= 1.1f;
	} 
	else{

		
		LoadDepthMap(ComposeReadDepthFilePath(depthData.GetView().GetID(), "dmap"), depthData.depthMap) ;
		LoadNormalMap(ComposeReadNormalFilePath(depthData.GetView().GetID(), "dmap"), depthData.normalMap) ;


		std::cout<<"read  :  "<< ComposeReadDepthFilePath(depthData.GetView().GetID(), "dmap") << std::endl;

		
		// ExportDepthMap(ComposeDepthFilePath(image.GetID(), "resize.png"),  depthData.resize_depthMap);
		// ExportNormalMap(ComposeNormalFilePath(image.GetID(), "resize.png"), depthData.resize_normalMap);

		
		cv::resize(depthData.depthMap, depthData.depthMap, depthData.depthMap.size(), 0 , 0 ,cv::INTER_CUBIC);
		cv::resize(depthData.normalMap, depthData.normalMap, depthData.depthMap.size(), 0 , 0 ,cv::INTER_CUBIC);

		for(int i = 0; i < depthData.depthMap.rows; i++){
			for(int j = 0; j < depthData.depthMap.cols; j++){
				float value = depthData.depthMap.at<float>(i,j);
				depthData.dMin = depthData.dMin > value ? value : depthData.dMin;
				depthData.dMax = depthData.dMax > value ? depthData.dMax : value;
			}
		}
		
		depthData.dMin *= 0.9f;
		depthData.dMax *= 1.1f;

		
		// ExportDepthMap(ComposeDepthFilePath(image.GetID(), "init.png"), depthData.depthMap);

	}
	

	
	




	#if TD_VERBOSE != TD_VERBOSE_OFF
	// save rough depth map as image
	if (g_nVerbosityLevel > 4) {
		ExportDepthMap(ComposeDepthFilePath(image.GetID(), "init.png"), depthData.depthMap);
		ExportNormalMap(ComposeDepthFilePath(image.GetID(), "init.normal.png"), depthData.normalMap);
		ExportPointCloud(ComposeDepthFilePath(image.GetID(), "init.ply"), *depthData.images.First().pImageData, depthData.depthMap, depthData.normalMap);
	}
	#endif

	DEBUG_ULTIMATE("Depth-map %3u roughly estimated from %u sparse points: %dx%d (%s)", image.GetID(), depthData.points.size(), image.image.width(), image.image.height(), TD_TIMER_GET_FMT().c_str());
	return true;
} // InitDepthMap
/*----------------------------------------------------------------*/

bool DepthMapsData::InitGraMap(DepthData& depthData)
{
	
	cv::Mat src,dst_x,dst_y;
		
	cv::cvtColor(depthData.images.First().pImageData->image, src, cv::COLOR_BGR2GRAY);

	
	cv::Sobel(src, dst_x, CV_16S, 1, 0, 3);
	cv::Sobel(src, dst_y, CV_16S, 0, 1, 3);
	cv::convertScaleAbs(dst_x, dst_x);
	cv::convertScaleAbs(dst_y, dst_y);
	
	
	cv::addWeighted(dst_x, 0.5, dst_y, 0.5, 0, depthData.graMap);
	// cv::imwrite("gra.jpg",depthData.graMap);
	// cv::imwrite("color.jpg",depthData.images.First().pImageData->image);
	// cv::imwrite("huidu.jpg",src);

	
	for(int i=0 ; i<depthData.graMap.width() ; i++)
	{
		for(int j=0 ; j<depthData.graMap.height() ; j++)
		{
			Point2i x ; x.x = i ; x.y = j ;
			if(depthData.graMap(x) < depthData.gra.gramin)
			{ depthData.gra.gramin = depthData.graMap(x) ; }
			if(depthData.graMap(x) > depthData.gra.gramax)
			{ depthData.gra.gramax = depthData.graMap(x) ; }
		}
	}

	
	for(int i=0 ; i<depthData.graMap.width() ; i++)
	{
		for(int j=0 ; j<depthData.graMap.height() ; j++)
		{
			Point2i x ; x.x = i ; x.y = j ;
			
			
			if( depthData.graMap(x) == depthData.gra.gramax )
			{
				
				if(x.x>10 && x.y>10 && x.x<depthData.graMap.width()-10 && x.y<depthData.graMap.height()-10 )
				{ depthData.gra.point = x ; }
			}
		}
	}
	std::cout<<std::endl;
	std::cout<<"depthData.gra.gramin = "<< depthData.gra.gramin <<std::endl;
	std::cout<<"depthData.gra.gramax = "<< depthData.gra.gramax <<std::endl;
	std::cout<<"depthData.gra.point = "<< depthData.gra.point <<std::endl;

	for(int i=0 ; i<depthData.diffMap.width() ; i++)
	{
		for(int j=0 ; j<depthData.diffMap.height() ; j++)
		{				
			Point2i x ; x.x = i ; x.y = j ;
			depthData.diffMap(x) = 0 ;
			depthData.ndiffMap(x) = 0 ;
			depthData.scoreGraMap(x) = 0 ;
		}
	}

}//InitGraMap


// initialize the confidence map (NCC score map) with the score of the current estimates
void* STCALL DepthMapsData::ScoreDepthMapTmp(void* arg)
{
	DepthEstimator& estimator = *((DepthEstimator*)arg);
	IDX idx;
	while ((idx=(IDX)Thread::safeInc(estimator.idxPixel)) < estimator.coords.GetSize()) {
		const ImageRef& x = estimator.coords[idx];		
		if (!estimator.PreparePixelPatch(x) || !estimator.FillPixelPatch()) {
			estimator.depthMap0(x) = 0;
			estimator.normalMap0(x) = Normal::ZERO;
			estimator.confMap0(x) = 2.f;
			continue;
		}
		Depth& depth = estimator.depthMap0(x);
		Normal& normal = estimator.normalMap0(x);
		const Normal viewDir(Cast<float>(static_cast<const Point3&>(estimator.X0)));
		if (!ISINSIDE(depth, estimator.dMin, estimator.dMax)) {
			// init with random values
			depth = estimator.RandomDepth(estimator.dMinSqr, estimator.dMaxSqr);
			normal = estimator.RandomNormal(viewDir);
		} else if (normal.dot(viewDir) >= 0) {
			// replace invalid normal with random values
			normal = estimator.RandomNormal(viewDir);
		}
		estimator.confMap0(x) = estimator.ScorePixel(depth, normal);
	}
	return NULL;
}
// run propagation and random refinement cycles
void* STCALL DepthMapsData::EstimateDepthMapTmp(void* arg)
{
	DepthEstimator& estimator = *((DepthEstimator*)arg);
	IDX idx;

	while ((idx=(IDX)Thread::safeInc(estimator.idxPixel)) < estimator.coords.GetSize())
		estimator.ProcessPixel(idx);	

	return NULL;
}
// remove all estimates with too big score and invert confidence map
void* STCALL DepthMapsData::EndDepthMapTmp(void* arg)
{
	DepthEstimator& estimator = *((DepthEstimator*)arg);
	IDX idx;
	const float fOptimAngle(FD2R(OPTDENSE::fOptimAngle));
	while ((idx=(IDX)Thread::safeInc(estimator.idxPixel)) < estimator.coords.GetSize()) {
		const ImageRef& x = estimator.coords[idx];
		ASSERT(estimator.depthMap0(x) >= 0);
		Depth& depth = estimator.depthMap0(x);
		float& conf = estimator.confMap0(x);
		// check if the score is good enough
		// and that the cross-estimates is close enough to the current estimate

		
		if (depth <= 0 || conf >= OPTDENSE::fNCCThresholdKeep) 
		{
			#if 1 // used if gap-interpolation is active
			conf = 0;
			estimator.normalMap0(x) = Normal::ZERO;
			#endif
			depth = 0;
		} 
		else 
		{
			#if 1
			// converted ZNCC [0-2] score, where 0 is best, to [0-1] confidence, where 1 is best
			conf = conf>=1.f ? 0.f : 1.f-conf;
			#else
			#if 1
			FOREACH(i, estimator.images)
				estimator.scores[i] = ComputeAngle<REAL,float>(estimator.image0.camera.TransformPointI2W(Point3(x,depth)).ptr(), estimator.image0.camera.C.ptr(), estimator.images[i].view.camera.C.ptr());
			#if DENSE_AGGNCC == DENSE_AGGNCC_NTH
			const float fCosAngle(estimator.scores.GetNth(estimator.idxScore));
			#elif DENSE_AGGNCC == DENSE_AGGNCC_MEAN
			const float fCosAngle(estimator.scores.mean());
			#elif DENSE_AGGNCC == DENSE_AGGNCC_MIN
			const float fCosAngle(estimator.scores.minCoeff());
			#else
			const float fCosAngle(estimator.idxScore ?
				std::accumulate(estimator.scores.begin(), &estimator.scores.PartialSort(estimator.idxScore), 0.f) / estimator.idxScore :
				*std::min_element(estimator.scores.cbegin(), estimator.scores.cend()));
			#endif
			const float wAngle(MINF(POW(ACOS(fCosAngle)/fOptimAngle,1.5f),1.f));
			#else
			const float wAngle(1.f);
			#endif
			#if 1
			conf = wAngle/MAXF(conf,1e-2f);
			#else
			conf = wAngle/(depth*SQUARE(MAXF(conf,1e-2f)));
			#endif
			#endif
		}
	}
	
	return NULL;
}

// estimate depth-map using propagation and random refinement with NCC score
// as in: "Accurate Multiple View 3D Reconstruction Using Patch-Based Stereo for Large-Scale Scenes", S. Shen, 2013
// The implementations follows closely the paper, although there are some changes/additions.
// Given two views of the same scene, we note as the "reference image" the view for which a depth-map is reconstructed, and the "target image" the other view.
// As a first step, the whole depth-map is approximated by interpolating between the available sparse points.
// Next, the depth-map is passed from top/left to bottom/right corner and the opposite sens for each of the next steps.
// For each pixel, first the current depth estimate is replaced with its neighbor estimates if the NCC score is better.
// Second, the estimate is refined by trying random estimates around the current depth and normal values, keeping the one with the best score.
// The estimation can be stopped at any point, and usually 2-3 iterations are enough for convergence.
// For each pixel, the depth and normal are scored by computing the NCC score between the patch in the reference image and the wrapped patch in the target image, as dictated by the homography matrix defined by the current values to be estimate.
// In order to ensure some smoothness while locally estimating each pixel, a bonus is added to the NCC score if the estimate for this pixel is close to the estimates for the neighbor pixels.
// Optionally, the occluded pixels can be detected by extending the described iterations to the target image and removing the estimates that do not have similar values in both views.
bool DepthMapsData::EstimateDepthMap(int it_external, IIndex idxImage)
{	
	TD_TIMER_STARTD();

	// initialize depth and normal maps
	DepthData& depthData(arrDepthData[idxImage]);
	ASSERT(depthData.images.GetSize() > 1 && !depthData.points.IsEmpty());
	const DepthData::ViewData& image(depthData.images.First());
	ASSERT(!image.image.empty() && !depthData.images[1].image.empty());
	const Image8U::Size size(image.image.size());

	const unsigned nMaxThreads(scene.nMaxThreads);

	
	if(it_external == 0)
	{
		depthData.depthMap.create(size); 
		depthData.depthMap.memset(0);
		depthData.normalMap.create(size);
		depthData.confMap.create(size);
		depthData.resize_depthMap.create(size);
		depthData.resize_normalMap.create(size);
		depthData.delta_c2pmax = 0;
		// initialize the depth-map
		
		if (OPTDENSE::nMinViewsTrustPoint < 2) {
			// compute depth range and initialize known depths
			const int nPixelArea(2); // half windows size around a pixel to be initialize with the known depth
			const Camera& camera = depthData.images.First().camera;
			depthData.dMin = FLT_MAX;
			depthData.dMax = 0;
			FOREACHPTR(pPoint, depthData.points) {
				const PointCloud::Point& X = scene.pointcloud.points[*pPoint];
				const Point3 camX(camera.TransformPointW2C(Cast<REAL>(X)));
				const ImageRef x(ROUND2INT(camera.TransformPointC2I(camX)));
				const float d((float)camX.z);
				const ImageRef sx(MAXF(x.x-nPixelArea,0), MAXF(x.y-nPixelArea,0));
				const ImageRef ex(MINF(x.x+nPixelArea,size.width-1), MINF(x.y+nPixelArea,size.height-1));
				for (int y=sx.y; y<=ex.y; ++y) {
					for (int x=sx.x; x<=ex.x; ++x) {
						depthData.depthMap(y,x) = d;
						depthData.normalMap(y,x) = Normal::ZERO;
					}
				}
				if (depthData.dMin > d)
					depthData.dMin = d;
				if (depthData.dMax < d)
					depthData.dMax = d;
			}
			depthData.dMin *= 0.9f;
			depthData.dMax *= 1.1f;
		} else {
			// compute rough estimates using the sparse point-cloud
			InitDepthMap(depthData);
		}

		
		depthData.graMap.create(size); 
		depthData.diffMap.create(size); 
		depthData.ndiffMap.create(size); 
		depthData.scoreGraMap.create(size); 
		InitGraMap(depthData);
		
	}
	
	// init integral images and index to image-ref map for the reference data
	
	#if DENSE_NCC == DENSE_NCC_WEIGHTED
	DepthEstimator::WeightMap weightMap0(size.area()-(size.width+1)*DepthEstimator::nSizeHalfWindow);
	#else
	Image64F imageSum0;
	cv::integral(image.image, imageSum0, CV_64F);
	#endif
	if (prevDepthMapSize != size) {
		BitMatrix mask;
		if (!OPTDENSE::nIgnoreMaskLabel.IsEmpty() && DepthEstimator::ImportIgnoreMask(*depthData.GetView().pImageData, depthData.depthMap.size(), mask, OPTDENSE::nIgnoreMaskLabel))
			depthData.ApplyIgnoreMask(mask);
		DepthEstimator::MapMatrix2ZigzagIdx(size, coords, mask, MAXF(64,(int)nMaxThreads*8));
		#if 0
		// show pixels to be processed
		Image8U cmask(size);
		cmask.memset(0);
		for (const DepthEstimator::MapRef& x: coords)
			cmask(x.y, x.x) = 255;
		cmask.Show("cmask");
		#endif
		if (mask.empty())
			prevDepthMapSize = size;
	}

	// init threads

	ASSERT(nMaxThreads > 0);
	cList<DepthEstimator> estimators;
	estimators.Reserve(nMaxThreads);
	cList<SEACAVE::Thread> threads;
	if (nMaxThreads > 1)
		threads.Resize(nMaxThreads-1); // current thread is also used
	volatile Thread::safe_t idxPixel;
	
	
	cv::medianBlur (depthData.depthMap , depthData.depthMap, 3);
	// cv::medianBlur (depthData.depthNormal , depthData.depthNormal, 3);
	
	if(0)
	{
		int width = depthData.depthMap.width();
		int height = depthData.depthMap.height();
		int midHalfwindow = 1 ;
		float depth_avg = 0 ;
		Normal normal_avg = Normal::ZERO ;
		int d_num = 0 ;
		int n_num = 0 ;
		for(int i=0 ; i<width ; i++)
		{
			for(int j=0 ; j<height ; j++)
			{
				
				Point2i x ; x.x = i ; x.y = j ;
				：
				if ( (x.x<width-midHalfwindow) && (x.y<height-midHalfwindow) && (x.x>midHalfwindow-1) && (x.y>midHalfwindow-1) ) 
				{
					
					for(int i2=x.x-midHalfwindow ; i2<x.x+midHalfwindow  ; i2++)
					{
						for(int j2=x.y-midHalfwindow ; j2<x.y+midHalfwindow ; j2++)
						{
							
							Point2i x1; x1.x=i2 ; x1.y=j2 ;
							if(depthData.depthMap(x1)!=0)
							{
								depth_avg += depthData.depthMap(x1) ; 
								d_num++ ;
							}
							if(depthData.normalMap(x1) != Normal::ZERO )
							{
								normal_avg += depthData.normalMap(x1) ;
								n_num++ ;
							}
						}
					}
				}
				depth_avg = depth_avg/d_num ;
				normal_avg.x = normal_avg.x/n_num ; normal_avg.y = normal_avg.y/n_num ; normal_avg.z = normal_avg.z/n_num ; 

				
				depthData.depthMap(x)=depth_avg ;
				// depthData.normalMap(x)=normal_avg ;

			}
		}
	}



	// initialize the reference confidence map (NCC score map) with the score of the current estimates
	
	{		
		// create working threads
		idxPixel = -1;
		ASSERT(estimators.IsEmpty());
		while (estimators.GetSize() < nMaxThreads)
			estimators.AddConstruct(0, it_external, depthData, arrDepthData, idxPixel,
				#if DENSE_NCC == DENSE_NCC_WEIGHTED
				weightMap0,
				#else
				imageSum0,
				#endif
				coords);
		ASSERT(estimators.GetSize() == threads.GetSize()+1);
		FOREACH(i, threads)
			threads[i].start(ScoreDepthMapTmp, &estimators[i]);
		ScoreDepthMapTmp(&estimators.Last());
		// wait for the working threads to close
		FOREACHPTR(pThread, threads)
			pThread->join();
		estimators.Release();
		#if TD_VERBOSE != TD_VERBOSE_OFF
		// save rough depth map as image
		if (g_nVerbosityLevel > 4) {
			ExportDepthMap(ComposeDepthFilePath(image.GetID(), "rough.png"), depthData.depthMap);
			ExportNormalMap(ComposeDepthFilePath(image.GetID(), "rough.normal.png"), depthData.normalMap);
			ExportPointCloud(ComposeDepthFilePath(image.GetID(), "rough.ply"), *depthData.images.First().pImageData, depthData.depthMap, depthData.normalMap);
		}
		#endif
	}



	// run propagation and random refinement cycles on the reference data
	// for (unsigned iter=0; iter<OPTDENSE::nEstimationIters; ++iter) 
	for (unsigned iter=0; iter<OPTDENSE::nEstimationIters; ++iter) 
	{
		//std::cout<<"before depthData.delta_c2pmax  is "<<depthData.delta_c2pmax<<std::endl;
		// create working threads
		idxPixel = -1;
		ASSERT(estimators.IsEmpty());
		while (estimators.GetSize() < nMaxThreads)
			estimators.AddConstruct(iter, it_external, depthData, arrDepthData, idxPixel,
				#if DENSE_NCC == DENSE_NCC_WEIGHTED
				weightMap0,
				#else
				imageSum0,
				#endif
				coords);
		ASSERT(estimators.GetSize() == threads.GetSize()+1);
		FOREACH(i, threads)
			threads[i].start(EstimateDepthMapTmp, &estimators[i]);
		EstimateDepthMapTmp(&estimators.Last());
		// wait for the working threads to close
		FOREACHPTR(pThread, threads)
			pThread->join();
		estimators.Release();
		//std::cout<<"depthData.delta_c2pmax  is  "<<depthData.delta_c2pmax<<std::endl;
		#if 1 && TD_VERBOSE != TD_VERBOSE_OFF
		// save intermediate depth map as image
		if (g_nVerbosityLevel > 4) {
			const String path(ComposeDepthFilePath(image.GetID(), "iter")+String::ToString(iter));
			ExportDepthMap(path+".png", depthData.depthMap);
			ExportNormalMap(path+".normal.png", depthData.normalMap);
			ExportPointCloud(path+".ply", *depthData.images.First().pImageData, depthData.depthMap, depthData.normalMap);
		}
		#endif
	}
	
	if(it_external == OPTDENSE::nEstimationIters_external-2)
	{
		if (OPTDENSE::nUseSemantic)
		{
			
			GenerateDepthPrior(depthData, coords);

			
			// GenerateFinalPrior(depthData, coords);

			// run propagation and random refinement cycles on the reference data
			
			#if 1
			for (unsigned iter = OPTDENSE::nEstimationIters; iter < OPTDENSE::nEstimationIters+2; ++iter) {
				// create working threads
				idxPixel = -1;
				ASSERT(estimators.IsEmpty());
				while (estimators.GetSize() < nMaxThreads)
					estimators.AddConstruct(iter, it_external, depthData, arrDepthData, idxPixel,
						#if DENSE_NCC == DENSE_NCC_WEIGHTED
						weightMap0,
						#else
						imageSum0,
						#endif
						coords);
				ASSERT(estimators.GetSize() == threads.GetSize() + 1);
				FOREACH(i, threads)
					threads[i].start(EstimateDepthMapTmp, &estimators[i]);
				EstimateDepthMapTmp(&estimators.Last());
				// wait for the working threads to close
				FOREACHPTR(pThread, threads)
					pThread->join();
				estimators.Release();
				#if 1 && TD_VERBOSE != TD_VERBOSE_OFF
				// save intermediate depth map as image
				if (g_nVerbosityLevel > 4) {
					const String path(ComposeDepthFilePath(image.GetID(), "iter") + String::ToString(iter));
					ExportDepthMap(path + ".png", depthData.depthMap);
					ExportNormalMap(path + ".normal.png", depthData.normalMap);
					ExportPointCloud(path + ".ply", *depthData.images.First().pImageData, depthData.depthMap, depthData.normalMap);
				}
				#endif
			}


			#endif

		}
	}

	// remove all estimates with too big score and invert confidence map
	
	if(it_external == OPTDENSE::nEstimationIters_external-1)
	{
		// create working threads
		idxPixel = -1;
		ASSERT(estimators.IsEmpty());
		while (estimators.GetSize() < nMaxThreads)
			estimators.AddConstruct(0, it_external, depthData, arrDepthData, idxPixel,
				#if DENSE_NCC == DENSE_NCC_WEIGHTED
				weightMap0,
				#else
				imageSum0,
				#endif
				coords);
		ASSERT(estimators.GetSize() == threads.GetSize()+1);
		FOREACH(i, threads)
			threads[i].start(EndDepthMapTmp, &estimators[i]);
		EndDepthMapTmp(&estimators.Last());
		// wait for the working threads to close
		FOREACHPTR(pThread, threads)
			pThread->join();
		estimators.Release();
	}
    
	
	ExportDepthMapByJetColormap(ComposeDepthFilePropagatedPath(depthData.GetView().GetID(), "png"), depthData.depthMap); 
	// ExportDepthDiffmap(ComposeDepthDiffPath(depthData.GetView().GetID(), "png"), depthData.diffMap); 
	// ExportDepthDiffmap(ComposeNormalDiffPath(depthData.GetView().GetID(), "png"), depthData.ndiffMap ); 
	// ExportDepthDiffmap(ComposeScoreGraPath(depthData.GetView().GetID(), "png"), depthData.scoreGraMap ); 
	// ExportNormalMap(ComposeNormalFilePath(depthData.GetView().GetID(), "png"), depthData.normalMap);


	DEBUG_EXTRA("Depth-map for image %3u %s: %dx%d (%s)", image.GetID(),
		depthData.images.GetSize() > 2 ?
			String::FormatString("estimated using %2u images", depthData.images.GetSize()-1).c_str() :
			String::FormatString("with image %3u estimated", depthData.images[1].GetID()).c_str(),
		size.width, size.height, TD_TIMER_GET_FMT().c_str());
	return true;
} // EstimateDepthMap
/*----------------------------------------------------------------*/





bool DepthMapsData::GenerateFinalPrior(DepthData& depthData, DepthEstimator::MapRefArr& coords)
{
	
	depthData.images.First().depthMapPrior = DepthMap(depthData.GetView().image.size());
	DepthMap& depthMap = depthData.images.First().depthMapPrior;
	depthData.images.First().normalMapPrior = NormalMap(depthData.GetView().image.size());
	NormalMap& normalMap = depthData.images.First().normalMapPrior;


	/**************************************************************meanshift*************************************************************/
	
	LoadDepthMap(ComposeMeanshiftDepthPriorsPath(depthData.GetView().GetID(), "dmap"), depthData.meanshiftpriors_depthMap) ;
	// LoadNormalMap(ComposeMeanshiftNormalPriorsPath(depthData.GetView().GetID(), "dmap"), depthData.meanshiftpriors_normalMap) ;

	
	// ExportDepthMap(ComposeDepthFilePath(depthData.GetView().GetID(), "dmap.png"), depthData.meanshiftpriors_depthMap);
	// ExportNormalMap(ComposeNormalFilePath(depthData.GetView().GetID(), "dmap.png"), depthData.meanshiftpriors_normalMap);

	
	resize(depthData.meanshiftpriors_depthMap , depthData.meanshiftpriors_depthMap , depthData.GetView().image.size());	
	// resize(depthData.meanshiftpriors_normalMap , depthData.meanshiftpriors_normalMap , depthData.GetView().image.size());	
	/*********************************************************************************************************************************** */


	
	LoadDepthMap(ComposeSuperDepthPriorsPath(depthData.GetView().GetID(), "dmap"), depthData.superpriors_depthMap) ;
	// LoadNormalMap(ComposeSuperNormalPriorsPath(depthData.GetView().GetID(), "dmap"), depthData.superpriors_normalMap) ;


	// ExportDepthMap(ComposeDepthFilePath(depthData.GetView().GetID(), "super.png"), depthData.superpriors_depthMap);
	// ExportNormalMap(ComposeNormalFilePath(depthData.GetView().GetID(), "super.png"), depthData.superpriors_normalMap);

	
	resize(depthData.superpriors_depthMap , depthData.superpriors_depthMap , depthData.GetView().image.size());	
	// resize(depthData.superpriors_normalMap , depthData.superpriors_normalMap , depthData.GetView().image.size());	
	// /*********************************************************************************************************************************** */


	
	IDX idx = -1;
	while (++idx < coords.GetSize())   
	{
		const ImageRef& coord = coords[idx];   
		depthMap(coord) = 0;                   
		normalMap(coord) = Normal::ZERO;	   
	}

	
	idx = -1;
	while (++idx < coords.GetSize())   
	{
		const ImageRef& coord = coords[idx];   
		if(depthData.meanshiftpriors_depthMap(coord) == 0 & depthData.superpriors_depthMap(coord) == 0){
			depthMap(coord) = 0;                  
			normalMap(coord) = Normal::ZERO;	  
			continue;
		}
		else if(depthData.meanshiftpriors_depthMap(coord) == 0){
			depthMap(coord) = depthData.superpriors_depthMap(coord) ;
			normalMap(coord) = depthData.superpriors_normalMap(coord) ;
		}
		else{
			depthMap(coord) = depthData.meanshiftpriors_depthMap(coord) ;
			normalMap(coord) = depthData.meanshiftpriors_normalMap(coord) ;
		}	
	}
	// while (++idx < coords.GetSize())   
	// {
	// 	const ImageRef& coord = coords[idx];   
	// 	if(depthData.meanshiftpriors_depthMap(coord) == 0){
	// 		depthMap(coord) = 0;                  
	// 		normalMap(coord) = Normal::ZERO;	  
	// 		continue;
	// 	}else{
	// 		depthMap(coord) = depthData.meanshiftpriors_depthMap(coord) ;
	// 		// depthMap(coord) = depthData.meanshiftpriors_normalMap(coord) ;
	// 	}	
	// }

	
	ExportDepthMap(ComposeDepthFilePath(depthData.GetView().GetID(), "finalprior.png"), depthMap);
	// ExportNormalMap(ComposeNormalFilePath(depthData.GetView().GetID(), "finalprior.png"), normalMap);
}









bool DepthMapsData::GenerateSuperDepthPrior(DepthData& depthData, const uint32_t label ,const int block_num)
{
	//#define PRIOR_DEBUG
	//typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel;
	typedef CGAL::Simple_cartesian<double> Kernel;
	typedef Kernel::Point_2 CGALPoint2;
	typedef Kernel::Point_3 CGALPoint;  
	typedef Kernel::Plane_3 CGALPlane;  
	typedef Kernel::Vector_3 CGALNormal;
	typedef std::pair<CGALPoint, CGALNormal> CGALPointWithNormal; 

	typedef Classification::Point_set_neighborhood<Kernel, std::vector<CGALPointWithNormal>, CGAL::First_of_pair_property_map<CGALPointWithNormal>> Neighborhood;
	typedef Classification::Local_eigen_analysis Local_eigen_analysis;

	typedef CGAL::Shape_detection::Efficient_RANSAC_traits<
		Kernel, 
		std::vector<CGALPointWithNormal>, 
		CGAL::First_of_pair_property_map<CGALPointWithNormal>, 
		CGAL::Second_of_pair_property_map<CGALPointWithNormal>> Traits;

	typedef CGAL::Shape_detection::Efficient_RANSAC<Traits> Efficient_ransac;
	typedef CGAL::Shape_detection::Plane<Traits> CGALRansacPlane;



	DepthMap& depthMap = depthData.superpriors_depthMap ;
	NormalMap& normalMap = depthData.superpriors_normalMap ;


	
	ImageRefArr regionPixels; // pixels of the region planar area 			

	
	std::vector<CGALPointWithNormal> ransacPointSamples;   
	std::vector<CGALPointWithNormal> filteredPointSamples; 

	#ifdef PRIOR_DEBUG
	std::ofstream stream(ComposeDepthFilePath(depthData.GetView().GetID(), "point.samples.txt").c_str());
	#endif

	
	IDX idx = -1;
	std::cout<<"label = "<< label <<std::endl ;
	std::cout<<"depthData.segmentLab[label].size() = "<<depthData.segmentLab[label].size()<<std::endl;
	int last_num = label+block_num;
	int rest_num = depthData.labelsNum - last_num;
	int num = 0 ;
	if(rest_num<block_num)
	{num=block_num+rest_num;}
	else
	{num=block_num;}
	for(int j=label ; j<label+num ; j++)
	{
		for(int i=1 ; i<depthData.segmentLab[j].size() ; i++ )
		{
			// std::cout<<"i="<<i<<std::endl ;
			const ImageRef coord = depthData.segmentLab[j][i] ;   		  
			depthMap(coord) = 0;                   
			normalMap(coord) = Normal::ZERO;	  


			regionPixels.Insert(coord);        	 

			// Try different conf threshold   
			
			if(depthData.depthMap(coord) != 0 && depthData.confMap(coord) > 0.6)
			{ 
				
				const Point3d point(depthData.images.First().camera.TransformPointI2W(Point3d(coord.x, coord.y, depthData.depthMap(coord))));  
				
				// Convert normal to object space  
				
				const Point3d N(depthData.images.First().camera.R.t() * Cast<REAL>(depthData.normalMap(coord)));
			
				
				ransacPointSamples.push_back(CGALPointWithNormal(CGALPoint(point.x, point.y, point.z), CGALNormal(N.x, N.y, N.z)));   

				#ifdef PRIOR_DEBUG
				stream << point.x << " " << point.y << " " << point.z << " " << N.x << " " << N.y << " " << N.z << " " << depthData.confMap(coord) << std::endl;
				#endif
				
			}
			
		}
	}

	std::cout<<"regionPixels.size() = "<<regionPixels.size()<<std::endl ;
	std::cout<<"ransacPointSamples.size() = "<<ransacPointSamples.size()<<std::endl ;

	#ifdef PRIOR_DEBUG
	stream.close();
	#endif

	Neighborhood neighborhood(ransacPointSamples, CGAL::First_of_pair_property_map<CGALPointWithNormal>());  //？？？
	Local_eigen_analysis eigen = Local_eigen_analysis::create_from_point_set(ransacPointSamples, CGAL::First_of_pair_property_map<CGALPointWithNormal>(), neighborhood.k_neighbor_query(10));   //？？？
				
	#ifdef PRIOR_DEBUG
	std::ofstream planarity(ComposeDepthFilePath(depthData.GetView().GetID(), "point.samples.planarity.txt").c_str());
	#endif

	
	for (int i = 0; i < ransacPointSamples.size(); i++)   
	{

		double c = eigen.eigenvalue(i)[0];  // x
		double b = eigen.eigenvalue(i)[1];  // y
		double a = eigen.eigenvalue(i)[2];  // z

		double p = 0;

		
		if ( (a >= 0 && b >= 0 && c >= 0) && a >= b && b >= c)
		{
			p = (b - c) / a;
		}
		// Try more flexible planarity threshold 
		// 0.5 for pipes
		if (p >= 0.3) 
		{
			
			filteredPointSamples.push_back(ransacPointSamples[i]);
		}

		#ifdef PRIOR_DEBUG
		planarity << ransacPointSamples[i].first.x() << " " << ransacPointSamples[i].first.y() << " " << ransacPointSamples[i].first.z() << " " << p << std::endl;
		#endif
	}

	#ifdef PRIOR_DEBUG
	planarity.close();
	#endif


	// considers n nearest neighbor points  
	
	const int nb_neighbors = 10; 

	// Estimate scale of the point set with average spacing 
	
	double average_spacing = CGAL::compute_average_spacing<CGAL::Sequential_tag>(filteredPointSamples, nb_neighbors, CGAL::parameters::point_map(CGAL::First_of_pair_property_map<CGALPointWithNormal>()));

	
	filteredPointSamples.erase(
		CGAL::remove_outliers(
			filteredPointSamples,
			nb_neighbors,
			CGAL::parameters::threshold_percent(100.). // No minimum percentage to remove
			threshold_distance(average_spacing).
			point_map(CGAL::First_of_pair_property_map<CGALPointWithNormal>())), 
		filteredPointSamples.end());

	#ifdef PRIOR_DEBUG
	std::ofstream pointsFilteredStream(ComposeDepthFilePath(depthData.GetView().GetID(), "point.samples.filtered.txt").c_str());

	for (std::vector<CGALPointWithNormal>::iterator it = filteredPointSamples.begin(); it != filteredPointSamples.end(); it++)
	{
		pointsFilteredStream << (*it).first.x() << " " << (*it).first.y() << " " << (*it).first.z() << " " << (*it).second.x() << " " << (*it).second.y() << " " << (*it).second.z() << std::endl;
	}

	pointsFilteredStream.close();
	#endif

	// Recompute average spacing after filtering. 
	
	average_spacing = CGAL::compute_average_spacing<CGAL::Sequential_tag>(filteredPointSamples, nb_neighbors, CGAL::parameters::point_map(CGAL::First_of_pair_property_map<CGALPointWithNormal>()));

	// Instantiate shape detection engine.
	
	Efficient_ransac ransac;

	// Provide input data.
	
	ransac.set_input(filteredPointSamples);  

	// Register planar shapes via template method.
	
	ransac.add_shape_factory<CGALRansacPlane>(); 

	Efficient_ransac::Parameters parameters;	
	
	// Set probability to miss the largest primitive at each iteration. 

	parameters.probability = OPTDENSE::ransacprobability;
	
	// Detect shapes with at least size / n points. 
	
	parameters.min_points = filteredPointSamples.size()/ OPTDENSE::fransacMinPointsDiv; //filteredPointSamples.size()/80 for ETH
	// parameters.min_points = filteredPointSamples.size()/ 20; 
	
	// Set maximum Euclidean distance between a point and a shape.   
	
	parameters.epsilon = average_spacing * OPTDENSE::fransacEpsilonMul;
	
	// Set maximum Euclidean distance between points to be clustered.     
	
	parameters.cluster_epsilon = average_spacing * OPTDENSE::fransacClusterMul;
	
	// Set maximum normal deviation (rad).
	// 0.9 < dot(surface_normal, point_normal);
	parameters.normal_threshold = 0.25; 
	
	// Detect registered shapes
	
	ransac.detect(parameters);

	std::cout << "epsilon: " << parameters.epsilon << std::endl;
	std::cout << "cluster: " << parameters.cluster_epsilon << std::endl;

	// Print number of detected shapes.
	std::cout << ransac.shapes().end() - ransac.shapes().begin() << " shapes detected with at least " << parameters.min_points << " points." << std::endl;

	Efficient_ransac::Shape_range shapes = ransac.shapes();

	Efficient_ransac::Shape_range::iterator it = shapes.begin();
	
	int planeIndex = 0;

	std::vector<std::pair<Planed, std::vector<CGALPoint2>>> planes;
	
	while (it != shapes.end())
	{
		CGALRansacPlane* shape = dynamic_cast<CGALRansacPlane*>(it->get());
		CGALPlane plane = static_cast<Kernel::Plane_3>(*shape);

		std::vector<CGALPoint> points; 

		#ifdef PRIOR_DEBUG
		const int pointsCount = (*it)->indices_of_assigned_points().size();
		std::ofstream planepoints((ComposeDepthFilePath(depthData.GetView().GetID(), "plane") + String::ToString(planeIndex) + ".ply").c_str());
		planepoints << "ply\nformat ascii 1.0\nelement vertex " << (pointsCount + 4) << "\nproperty float x\nproperty float y\nproperty float z\nproperty float plane_index\nelement face 2\nproperty list uchar int vertex_indices\nend_header\n";
		#endif 

		for (std::vector<std::size_t>::const_iterator index = (*it)->indices_of_assigned_points().begin(); index != (*it)->indices_of_assigned_points().end(); index++)
		{	
			#ifdef PRIOR_DEBUG	
			planepoints << filteredPointSamples[(*index)].first.x() << " " << filteredPointSamples[(*index)].first.y() << " " << filteredPointSamples[(*index)].first.z() << " " << planeIndex << std::endl;
			#endif

			points.push_back(filteredPointSamples[(*index)].first);
		}

		CGALPoint center = centroid(points.begin(), points.end()); //CGALPoint((bbox.xmin() + bbox.xmax()) / 2, (bbox.ymin() + bbox.ymax()) / 2, (bbox.zmin() + bbox.zmax()) / 2);
		
		Planed finalPlane(Eigen::Vector3d(plane.orthogonal_vector().x(), plane.orthogonal_vector().y(), plane.orthogonal_vector().z()), (Point3d(center.x(), center.y(), center.z())));			
		
		cv::Vec3d xAxis = cv::normalize(cv::Vec3d(plane.base1().x(), plane.base1().y(), plane.base1().z()));
		cv::Vec3d yAxis = cv::normalize(cv::Vec3d(plane.base2().x(), plane.base2().y(), plane.base2().z()));
		cv::Vec3d zAxis = cv::normalize(cv::Vec3d(plane.orthogonal_vector().x(), plane.orthogonal_vector().y(), plane.orthogonal_vector().z()));

		
		/*Point3d base_1 = (Point3d(plane.base1().x(), plane.base1().y(), plane.base1().z()));
		Point3d base_2 = (Point3d(plane.base2().x(), plane.base2().y(), plane.base2().z()));*/

		double minX = std::numeric_limits<float>::max();
		double maxX = -std::numeric_limits<float>::max();
		double minY = std::numeric_limits<float>::max();
		double maxY = -std::numeric_limits<float>::max();
		
		// project each point to the plane so that all of them have the same z
		// calculate min max point
		// calculate dimensions
	
		for (size_t i = 0; i < points.size(); i++)
		{				
			CGALPoint projectedPoint = plane.projection(points[i]);

			double x = xAxis.dot(cv::Vec3d(projectedPoint.x(), projectedPoint.y(), projectedPoint.z()) - cv::Vec3d(center.x(), center.y(), center.z())); //double x = Side	| (point - Center);
			double y = yAxis.dot(cv::Vec3d(projectedPoint.x(), projectedPoint.y(), projectedPoint.z()) - cv::Vec3d(center.x(), center.y(), center.z())); //double y = Up | (point - Center);									

			minX = std::min(x, minX);
			minY = std::min(y, minY);

			maxX = std::max(x, maxX);
			maxY = std::max(y, maxY);														
		}

		Point3d upperRight = cv::Vec3d(center.x(), center.y(), center.z()) + xAxis * maxX + yAxis * maxY;
		Point3d upperLeft = cv::Vec3d(center.x(), center.y(), center.z()) + xAxis * minX + yAxis * maxY;
		Point3d lowerRight = cv::Vec3d(center.x(), center.y(), center.z()) + xAxis * maxX + yAxis * minY;
		Point3d lowerLeft = cv::Vec3d(center.x(), center.y(), center.z()) + xAxis * minX + yAxis * minY;

		Point2 upperRight_2D = depthData.images.First().camera.TransformPointW2I(Point3(upperRight[0], upperRight[1], upperRight[2]));
		Point2 upperLeft_2D = depthData.images.First().camera.TransformPointW2I(Point3(upperLeft[0], upperLeft[1], upperLeft[2]));
		Point2 lowerRight_2D = depthData.images.First().camera.TransformPointW2I(Point3(lowerRight[0], lowerRight[1], lowerRight[2]));
		Point2 lowerLeft_2D = depthData.images.First().camera.TransformPointW2I(Point3(lowerLeft[0], lowerLeft[1], lowerLeft[2]));

		std::vector<CGALPoint2> BB = { 
			CGALPoint2(lowerLeft_2D.x, lowerLeft_2D.y), 
			CGALPoint2(lowerRight_2D.x, lowerRight_2D.y), 
			CGALPoint2(upperRight_2D.x, upperRight_2D.y),
			CGALPoint2(upperLeft_2D.x, upperLeft_2D.y) 
		};				

		#ifdef PRIOR_DEBUG				
		planepoints << lowerLeft.x << " " << lowerLeft.y << " " << lowerLeft.z << " " << planeIndex << std::endl;
		planepoints << upperLeft.x << " " << upperLeft.y << " " << upperLeft.z << " " << planeIndex << std::endl;
		planepoints << upperRight.x << " " << upperRight.y << " " << upperRight.z << " " << planeIndex << std::endl;			
		planepoints << lowerRight.x << " " << lowerRight.y << " " << lowerRight.z << " " << planeIndex << std::endl;				
		planepoints << "3 " << pointsCount << " " << (pointsCount + 2) << " " << (pointsCount + 1) << std::endl;
		planepoints << "3 " << pointsCount << " " << (pointsCount + 3) << " " << (pointsCount + 2) << std::endl;
		planepoints.close();
		#endif

		// Use Bounding Box
		std::pair<Planed, std::vector<CGALPoint2>> planeAndBBox(finalPlane, BB);

		planes.push_back(planeAndBBox);
		points.clear();

		planeIndex++;
		it++;				
	}

	Point3d origin(depthData.images.First().camera.C.x, depthData.images.First().camera.C.y, depthData.images.First().camera.C.z);

	#ifdef PRIOR_DEBUG
	std::ofstream intersections(ComposeDepthFilePath(depthData.GetView().GetID(), "intersections.txt").c_str());
	#endif

	// Iterate through region pixels and do the ray tracing
	
	idx = -1;
	while (++idx < regionPixels.GetSize()) 
	{
		const ImageRef& coord = regionPixels[idx];

		Point3d grid(depthData.images.First().camera.TransformPointI2W(Point3(coord.x, coord.y, 1)));
		Point3d test(depthData.images.First().camera.TransformPointI2W(Point3(coord.x, coord.y, depthData.depthMap(coord))));

		int planeId = -1;
		float minDistance = std::numeric_limits<float>::max();
		
		Point3d direction = grid - origin;
		const Ray3d ray(origin, direction);

		for (int i = 0; i < planes.size(); i++)
		{										
			// check if inside the BB
			bool isInside = CGAL::bounded_side_2(planes[i].second.begin(), planes[i].second.end(), CGALPoint2(coord.x, coord.y), Kernel()) == CGAL::ON_BOUNDED_SIDE;					
			
			if (isInside)
			{
				//float distance = ray.IntersectsDist(planes[i].first); //planes[i].first.DistanceAbs(test);
				float distance = planes[i].first.DistanceAbs(test);

				if (distance >= 0 && distance < minDistance) 
				{
					minDistance = distance;
					planeId = i;				
				}
			}
		}				

		if (planeId >= 0)
		{
			const Point3d intersection(ray.Intersects(planes[planeId].first));
			
			#ifdef PRIOR_DEBUG
			intersections << grid.x << " " << grid.y << " " << grid.z << " " << planeId << std::endl;
			#endif

			depthMap(coord) = depthData.images.First().camera.TransformPointW2I3(intersection).z;
			const Normal planeNormal(planes[planeId].first.m_vN.x(), planes[planeId].first.m_vN.y(), planes[planeId].first.m_vN.z()); // Normal in object space	
			const Normal N(depthData.images.First().camera.R * Cast<REAL>(planeNormal)); // Convert back in image space

			normalMap(coord) = N;
		}				
	}

	#ifdef PRIOR_DEBUG
	intersections.close();
	#endif

	return true;
}

//GenerateSuperDepthPrior


bool DepthMapsData::GenerateDepthPrior(DepthData& depthData, DepthEstimator::MapRefArr& coords)
{
	//#define PRIOR_DEBUG
	//typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel;
	typedef CGAL::Simple_cartesian<double> Kernel;
	typedef Kernel::Point_2 CGALPoint2;
	typedef Kernel::Point_3 CGALPoint; 
	typedef Kernel::Plane_3 CGALPlane;  
	typedef Kernel::Vector_3 CGALNormal;
	typedef std::pair<CGALPoint, CGALNormal> CGALPointWithNormal; 

	typedef Classification::Point_set_neighborhood<Kernel, std::vector<CGALPointWithNormal>, CGAL::First_of_pair_property_map<CGALPointWithNormal>> Neighborhood;
	typedef Classification::Local_eigen_analysis Local_eigen_analysis;

	typedef CGAL::Shape_detection::Efficient_RANSAC_traits<
		Kernel, 
		std::vector<CGALPointWithNormal>, 
		CGAL::First_of_pair_property_map<CGALPointWithNormal>, 
		CGAL::Second_of_pair_property_map<CGALPointWithNormal>> Traits;

	typedef CGAL::Shape_detection::Efficient_RANSAC<Traits> Efficient_ransac;
	typedef CGAL::Shape_detection::Plane<Traits> CGALRansacPlane;

	std::cout << "\nLoading Label Mask Image: " << (*depthData.GetView().pImageData).maskName << std::endl;

	
	if (depthData.labels.Load((*depthData.GetView().pImageData).maskName)) 
	{
		 
		if (depthData.labels.size() != depthData.GetView().image.size()) 
		{
			
			cv::resize(depthData.labels, depthData.labels, depthData.GetView().image.size(), 0, 0, cv::INTER_NEAREST);
	
			
			// cv::imwrite("/home/xx/Code_hn/openMVS_seman_priors/run/output_pipes/mvs/labels.png",depthData.labels);
		}
		
		
		depthData.images.First().depthMapPrior = DepthMap(depthData.GetView().image.size());
		DepthMap& depthMap = depthData.images.First().depthMapPrior;
		depthData.images.First().normalMapPrior = NormalMap(depthData.GetView().image.size());
		NormalMap& normalMap = depthData.images.First().normalMapPrior;

		
		if (!depthData.labels.empty())
		{
			
			ImageRefArr regionPixels; // pixels of the region planar area 			

			
			std::vector<CGALPointWithNormal> ransacPointSamples;   
			std::vector<CGALPointWithNormal> filteredPointSamples; 

			#ifdef PRIOR_DEBUG
			std::ofstream stream(ComposeDepthFilePath(depthData.GetView().GetID(), "point.samples.txt").c_str());
			#endif

			
			
			
			IDX idx = -1;
			while (++idx < coords.GetSize())   
			{
				const ImageRef& coord = coords[idx];   
				depthMap(coord) = 0;                   
				normalMap(coord) = Normal::ZERO;	   

				
				if (depthData.labels(coord) == 255)
				{
					regionPixels.Insert(coord);        	 

					// Try different conf threshold   
					
					if(depthData.depthMap(coord) != 0 && depthData.confMap(coord) < 0.18) 
					{
						
						const Point3d point(depthData.images.First().camera.TransformPointI2W(Point3d(coord.x, coord.y, depthData.depthMap(coord))));  
						
						// Convert normal to object space  
						
						const Point3d N(depthData.images.First().camera.R.t() * Cast<REAL>(depthData.normalMap(coord)));

						
						ransacPointSamples.push_back(CGALPointWithNormal(CGALPoint(point.x, point.y, point.z), CGALNormal(N.x, N.y, N.z)));   

						#ifdef PRIOR_DEBUG
						stream << point.x << " " << point.y << " " << point.z << " " << N.x << " " << N.y << " " << N.z << " " << depthData.confMap(coord) << std::endl;
						#endif
					}
				}
			}

			#ifdef PRIOR_DEBUG
			stream.close();
			#endif

			Neighborhood neighborhood(ransacPointSamples, CGAL::First_of_pair_property_map<CGALPointWithNormal>());  //？？？
			Local_eigen_analysis eigen = Local_eigen_analysis::create_from_point_set(ransacPointSamples, CGAL::First_of_pair_property_map<CGALPointWithNormal>(), neighborhood.k_neighbor_query(10));   //？？？
						
			#ifdef PRIOR_DEBUG
			std::ofstream planarity(ComposeDepthFilePath(depthData.GetView().GetID(), "point.samples.planarity.txt").c_str());
			#endif

			
			for (int i = 0; i < ransacPointSamples.size(); i++)   
			{

				double c = eigen.eigenvalue(i)[0];  // x
				double b = eigen.eigenvalue(i)[1];  // y
				double a = eigen.eigenvalue(i)[2];  // z

				double p = 0;

				
				if ( (a >= 0 && b >= 0 && c >= 0) && a >= b && b >= c)
				{
					p = (b - c) / a;
				}
				// Try more flexible planarity threshold 
				// 0.5 for pipes
				if (p >= 0.3) 
				{
					
					filteredPointSamples.push_back(ransacPointSamples[i]);
				}

				#ifdef PRIOR_DEBUG
				planarity << ransacPointSamples[i].first.x() << " " << ransacPointSamples[i].first.y() << " " << ransacPointSamples[i].first.z() << " " << p << std::endl;
				#endif
			}

			#ifdef PRIOR_DEBUG
			planarity.close();
			#endif


			// considers n nearest neighbor points  
			
			const int nb_neighbors = 10; 

			// Estimate scale of the point set with average spacing 
			
			double average_spacing = CGAL::compute_average_spacing<CGAL::Sequential_tag>(filteredPointSamples, nb_neighbors, CGAL::parameters::point_map(CGAL::First_of_pair_property_map<CGALPointWithNormal>()));

			
			filteredPointSamples.erase(
				CGAL::remove_outliers(
					filteredPointSamples,
					nb_neighbors,
					CGAL::parameters::threshold_percent(100.). // No minimum percentage to remove
					threshold_distance(average_spacing).
					point_map(CGAL::First_of_pair_property_map<CGALPointWithNormal>())), 
				filteredPointSamples.end());

			#ifdef PRIOR_DEBUG
			std::ofstream pointsFilteredStream(ComposeDepthFilePath(depthData.GetView().GetID(), "point.samples.filtered.txt").c_str());

			for (std::vector<CGALPointWithNormal>::iterator it = filteredPointSamples.begin(); it != filteredPointSamples.end(); it++)
			{
				pointsFilteredStream << (*it).first.x() << " " << (*it).first.y() << " " << (*it).first.z() << " " << (*it).second.x() << " " << (*it).second.y() << " " << (*it).second.z() << std::endl;
			}

			pointsFilteredStream.close();
			#endif

			// Recompute average spacing after filtering. 
			
			average_spacing = CGAL::compute_average_spacing<CGAL::Sequential_tag>(filteredPointSamples, nb_neighbors, CGAL::parameters::point_map(CGAL::First_of_pair_property_map<CGALPointWithNormal>()));

			// Instantiate shape detection engine.
			
			Efficient_ransac ransac;

			// Provide input data.
			
			ransac.set_input(filteredPointSamples);  

			// Register planar shapes via template method.
			
			ransac.add_shape_factory<CGALRansacPlane>(); 

			Efficient_ransac::Parameters parameters;	
			
			// Set probability to miss the largest primitive at each iteration. 
			
			parameters.probability = OPTDENSE::ransacprobability ;
			
			// Detect shapes with at least size / n points. 
			
			parameters.min_points = filteredPointSamples.size()/ OPTDENSE::fransacMinPointsDiv; //filteredPointSamples.size()/80 for ETH
			
			// Set maximum Euclidean distance between a point and a shape.   
			
			parameters.epsilon = average_spacing * OPTDENSE::fransacEpsilonMul;
			
			// Set maximum Euclidean distance between points to be clustered.     
			
			parameters.cluster_epsilon = average_spacing * OPTDENSE::fransacClusterMul;
			
			// Set maximum normal deviation (rad).
			// 0.9 < dot(surface_normal, point_normal);
			parameters.normal_threshold = 0.25;    
			
			// Detect registered shapes
			
			ransac.detect(parameters);     

			std::cout << "epsilon: " << parameters.epsilon << std::endl;
			std::cout << "cluster: " << parameters.cluster_epsilon << std::endl;

			// Print number of detected shapes.
			
			std::cout << ransac.shapes().end() - ransac.shapes().begin() << " shapes detected with at least " << parameters.min_points << " points." << std::endl;

			Efficient_ransac::Shape_range shapes = ransac.shapes();

			Efficient_ransac::Shape_range::iterator it = shapes.begin();
			
			int planeIndex = 0;

			std::vector<std::pair<Planed, std::vector<CGALPoint2>>> planes;
			
			while (it != shapes.end())
			{
				CGALRansacPlane* shape = dynamic_cast<CGALRansacPlane*>(it->get());
				CGALPlane plane = static_cast<Kernel::Plane_3>(*shape);

				std::vector<CGALPoint> points; 

				#ifdef PRIOR_DEBUG
				const int pointsCount = (*it)->indices_of_assigned_points().size();
				std::ofstream planepoints((ComposeDepthFilePath(depthData.GetView().GetID(), "plane") + String::ToString(planeIndex) + ".ply").c_str());
				planepoints << "ply\nformat ascii 1.0\nelement vertex " << (pointsCount + 4) << "\nproperty float x\nproperty float y\nproperty float z\nproperty float plane_index\nelement face 2\nproperty list uchar int vertex_indices\nend_header\n";
				#endif 

				for (std::vector<std::size_t>::const_iterator index = (*it)->indices_of_assigned_points().begin(); index != (*it)->indices_of_assigned_points().end(); index++)
				{	
					#ifdef PRIOR_DEBUG	
					planepoints << filteredPointSamples[(*index)].first.x() << " " << filteredPointSamples[(*index)].first.y() << " " << filteredPointSamples[(*index)].first.z() << " " << planeIndex << std::endl;
					#endif

					points.push_back(filteredPointSamples[(*index)].first);
				}

				CGALPoint center = centroid(points.begin(), points.end()); //CGALPoint((bbox.xmin() + bbox.xmax()) / 2, (bbox.ymin() + bbox.ymax()) / 2, (bbox.zmin() + bbox.zmax()) / 2);
				
				Planed finalPlane(Eigen::Vector3d(plane.orthogonal_vector().x(), plane.orthogonal_vector().y(), plane.orthogonal_vector().z()), (Point3d(center.x(), center.y(), center.z())));			
				
				cv::Vec3d xAxis = cv::normalize(cv::Vec3d(plane.base1().x(), plane.base1().y(), plane.base1().z()));
				cv::Vec3d yAxis = cv::normalize(cv::Vec3d(plane.base2().x(), plane.base2().y(), plane.base2().z()));
				cv::Vec3d zAxis = cv::normalize(cv::Vec3d(plane.orthogonal_vector().x(), plane.orthogonal_vector().y(), plane.orthogonal_vector().z()));

				
				/*Point3d base_1 = (Point3d(plane.base1().x(), plane.base1().y(), plane.base1().z()));
				Point3d base_2 = (Point3d(plane.base2().x(), plane.base2().y(), plane.base2().z()));*/

				double minX = std::numeric_limits<float>::max();
				double maxX = -std::numeric_limits<float>::max();
				double minY = std::numeric_limits<float>::max();
				double maxY = -std::numeric_limits<float>::max();
				
				// project each point to the plane so that all of them have the same z
				// calculate min max point
				// calculate dimensions
			
				for (size_t i = 0; i < points.size(); i++)
				{				
					CGALPoint projectedPoint = plane.projection(points[i]);

					double x = xAxis.dot(cv::Vec3d(projectedPoint.x(), projectedPoint.y(), projectedPoint.z()) - cv::Vec3d(center.x(), center.y(), center.z())); //double x = Side	| (point - Center);
					double y = yAxis.dot(cv::Vec3d(projectedPoint.x(), projectedPoint.y(), projectedPoint.z()) - cv::Vec3d(center.x(), center.y(), center.z())); //double y = Up | (point - Center);									

					minX = std::min(x, minX);
					minY = std::min(y, minY);

					maxX = std::max(x, maxX);
					maxY = std::max(y, maxY);														
				}

				Point3d upperRight = cv::Vec3d(center.x(), center.y(), center.z()) + xAxis * maxX + yAxis * maxY;
				Point3d upperLeft = cv::Vec3d(center.x(), center.y(), center.z()) + xAxis * minX + yAxis * maxY;
				Point3d lowerRight = cv::Vec3d(center.x(), center.y(), center.z()) + xAxis * maxX + yAxis * minY;
				Point3d lowerLeft = cv::Vec3d(center.x(), center.y(), center.z()) + xAxis * minX + yAxis * minY;

				Point2 upperRight_2D = depthData.images.First().camera.TransformPointW2I(Point3(upperRight[0], upperRight[1], upperRight[2]));
				Point2 upperLeft_2D = depthData.images.First().camera.TransformPointW2I(Point3(upperLeft[0], upperLeft[1], upperLeft[2]));
				Point2 lowerRight_2D = depthData.images.First().camera.TransformPointW2I(Point3(lowerRight[0], lowerRight[1], lowerRight[2]));
				Point2 lowerLeft_2D = depthData.images.First().camera.TransformPointW2I(Point3(lowerLeft[0], lowerLeft[1], lowerLeft[2]));

				std::vector<CGALPoint2> BB = { 
					CGALPoint2(lowerLeft_2D.x, lowerLeft_2D.y), 
					CGALPoint2(lowerRight_2D.x, lowerRight_2D.y), 
					CGALPoint2(upperRight_2D.x, upperRight_2D.y),
					CGALPoint2(upperLeft_2D.x, upperLeft_2D.y) 
				};				

				#ifdef PRIOR_DEBUG				
				planepoints << lowerLeft.x << " " << lowerLeft.y << " " << lowerLeft.z << " " << planeIndex << std::endl;
				planepoints << upperLeft.x << " " << upperLeft.y << " " << upperLeft.z << " " << planeIndex << std::endl;
				planepoints << upperRight.x << " " << upperRight.y << " " << upperRight.z << " " << planeIndex << std::endl;			
				planepoints << lowerRight.x << " " << lowerRight.y << " " << lowerRight.z << " " << planeIndex << std::endl;				
				planepoints << "3 " << pointsCount << " " << (pointsCount + 2) << " " << (pointsCount + 1) << std::endl;
				planepoints << "3 " << pointsCount << " " << (pointsCount + 3) << " " << (pointsCount + 2) << std::endl;
				planepoints.close();
				#endif

				// Use Bounding Box
				std::pair<Planed, std::vector<CGALPoint2>> planeAndBBox(finalPlane, BB);

				planes.push_back(planeAndBBox);
				points.clear();

				planeIndex++;
				it++;				
			}

			Point3d origin(depthData.images.First().camera.C.x, depthData.images.First().camera.C.y, depthData.images.First().camera.C.z);

			#ifdef PRIOR_DEBUG
			std::ofstream intersections(ComposeDepthFilePath(depthData.GetView().GetID(), "intersections.txt").c_str());
			#endif

			// Iterate through region pixels and do the ray tracing
			
			idx = -1;
			while (++idx < regionPixels.GetSize()) 
			{
				const ImageRef& coord = regionPixels[idx];

				Point3d grid(depthData.images.First().camera.TransformPointI2W(Point3(coord.x, coord.y, 1)));
				Point3d test(depthData.images.First().camera.TransformPointI2W(Point3(coord.x, coord.y, depthData.depthMap(coord))));

				int planeId = -1;
				float minDistance = std::numeric_limits<float>::max();
				
				Point3d direction = grid - origin;
				const Ray3d ray(origin, direction);

				for (int i = 0; i < planes.size(); i++)
				{										
					// check if inside the BB
					bool isInside = CGAL::bounded_side_2(planes[i].second.begin(), planes[i].second.end(), CGALPoint2(coord.x, coord.y), Kernel()) == CGAL::ON_BOUNDED_SIDE;					
					
					if (isInside)
					{
						//float distance = ray.IntersectsDist(planes[i].first); //planes[i].first.DistanceAbs(test);
						float distance = planes[i].first.DistanceAbs(test);

						if (distance >= 0 && distance < minDistance) 
						{
							minDistance = distance;
							planeId = i;				
						}
					}
				}				

				if (planeId >= 0)
				{
					const Point3d intersection(ray.Intersects(planes[planeId].first));
					
					#ifdef PRIOR_DEBUG
					intersections << grid.x << " " << grid.y << " " << grid.z << " " << planeId << std::endl;
					#endif

					depthMap(coord) = depthData.images.First().camera.TransformPointW2I3(intersection).z;
					const Normal planeNormal(planes[planeId].first.m_vN.x(), planes[planeId].first.m_vN.y(), planes[planeId].first.m_vN.z()); // Normal in object space	
					const Normal N(depthData.images.First().camera.R * Cast<REAL>(planeNormal)); // Convert back in image space

					normalMap(coord) = N;
				}				
			}

			#ifdef PRIOR_DEBUG
			intersections.close();
			#endif
			

			
			SaveDepthMap(ComposeMeanshiftDepthPriorsPath(depthData.GetView().GetID(), "dmap"), depthMap);
			SaveNormalMap(ComposeMeanshiftNormalPriorsPath(depthData.GetView().GetID(), "dmap"), normalMap);

			
			// LoadDepthMap(ComposeMeanshiftDepthPriorsPath(depthData.GetView().GetID(), "dmap"), depthMap) ;
			// LoadNormalMap(ComposeMeanshiftNormalPriorsPath(depthData.GetView().GetID(), "dmap"), normalMap) ;

			
			ExportDepthMap(ComposeDepthFilePath(depthData.GetView().GetID(), "prior.png"), depthMap);
			ExportNormalMap(ComposeNormalFilePath(depthData.GetView().GetID(), "prior.normal.png"), normalMap);

			
			// ExportDepthMapByJetColormap(ComposeDepthFilePropagatedPath(depthData.GetView().GetID(), "tworpropagete.png"), depthData.depthMap);
			
			
		
		} 
	}

	return true;
}

// filter out small depth segments from the given depth map
bool DepthMapsData::RemoveSmallSegments(DepthData& depthData)
{
	// std::cout<<"2222222222"<<std::endl;
	#if 0
	const float fDepthDiffThreshold(OPTDENSE::fDepthDiffThreshold*0.7f);
	unsigned speckle_size = OPTDENSE::nSpeckleSize;
	DepthMap& depthMap = depthData.depthMap;
	NormalMap& normalMap = depthData.normalMap;
	ConfidenceMap& confMap = depthData.confMap;
	ASSERT(!depthMap.empty());
	const ImageRef size(depthMap.size());

	// allocate memory on heap for dynamic programming arrays
	TImage<bool> done_map(size, false);
	CAutoPtrArr<ImageRef> seg_list(new ImageRef[size.x*size.y]);
	unsigned seg_list_count;
	unsigned seg_list_curr;
	ImageRef neighbor[4];

	// for all pixels do
	for (int u=0; u<size.x; ++u) {
		for (int v=0; v<size.y; ++v) {
			// if the first pixel in this segment has been already processed => skip
			if (done_map(v,u))
				continue;

			// init segment list (add first element
			// and set it to be the next element to check)
			seg_list[0] = ImageRef(u,v);
			seg_list_count = 1;
			seg_list_curr  = 0;

			// add neighboring segments as long as there
			// are none-processed pixels in the seg_list;
			// none-processed means: seg_list_curr<seg_list_count
			while (seg_list_curr < seg_list_count) {
				// get address of current pixel in this segment
				const ImageRef addr_curr(seg_list[seg_list_curr]);
				const Depth& depth_curr = depthMap(addr_curr);

				if (depth_curr>0) {
					// fill list with neighbor positions
					neighbor[0] = ImageRef(addr_curr.x-1, addr_curr.y  );
					neighbor[1] = ImageRef(addr_curr.x+1, addr_curr.y  );
					neighbor[2] = ImageRef(addr_curr.x  , addr_curr.y-1);
					neighbor[3] = ImageRef(addr_curr.x  , addr_curr.y+1);

					// for all neighbors do
					for (int i=0; i<4; ++i) {
						// get neighbor pixel address
						const ImageRef& addr_neighbor(neighbor[i]);
						// check if neighbor is inside image
						if (addr_neighbor.x>=0 && addr_neighbor.y>=0 && addr_neighbor.x<size.x && addr_neighbor.y<size.y) {
							// check if neighbor has not been added yet
							bool& done = done_map(addr_neighbor);
							if (!done) {
								// check if the neighbor is valid and similar to the current pixel
								// (belonging to the current segment)
								const Depth& depth_neighbor = depthMap(addr_neighbor);
								if (depth_neighbor>0 && IsDepthSimilar(depth_curr, depth_neighbor, fDepthDiffThreshold)) {
									// add neighbor coordinates to segment list
									seg_list[seg_list_count++] = addr_neighbor;
									// set neighbor pixel in done_map to "done"
									// (otherwise a pixel may be added 2 times to the list, as
									//  neighbor of one pixel and as neighbor of another pixel)
									done = true;
								}
							}
						}
					}
				}

				// set current pixel in seg_list to "done"
				++seg_list_curr;

				// set current pixel in done_map to "done"
				done_map(addr_curr) = true;
			} // end: while (seg_list_curr < seg_list_count)

			// if segment NOT large enough => invalidate pixels
			if (seg_list_count < speckle_size) {
				// for all pixels in current segment invalidate pixels
				for (unsigned i=0; i<seg_list_count; ++i) {
					depthMap(seg_list[i]) = 0;
					if (!normalMap.empty()) normalMap(seg_list[i]) = Normal::ZERO;
					if (!confMap.empty()) confMap(seg_list[i]) = 0;
				}
			}
		}
	}
	#else
	TD_TIMER_STARTD();

	struct Proj {
		union {
			uint32_t idxPixel;
			struct {
				uint16_t x, y; // image pixel coordinates
			};
		};
		inline Proj() {}
		inline Proj(uint32_t _idxPixel) : idxPixel(_idxPixel) {}
		inline Proj(const ImageRef& ir) : x(ir.x), y(ir.y) {}
		inline ImageRef GetCoord() const { return ImageRef(x,y); }
	};
	typedef SEACAVE::cList<Proj,const Proj&,0,4,uint32_t> ProjArr;
	typedef SEACAVE::cList<ProjArr,const ProjArr&,1,65536> ProjsArr;

	// find best connected images
	IndexScoreArr connections(0, scene.images.GetSize());
	size_t nPointsEstimate(0);
	bool bNormalMap(true);
	FOREACH(i, scene.images) {
		DepthData& depthData = arrDepthData[i];
		if (!depthData.IsValid())
			continue;
		if (depthData.IncRef(ComposeDepthFilePath(depthData.GetView().GetID(), "dmap")) == 0)
			return true;
		ASSERT(!depthData.IsEmpty());
		IndexScore& connection = connections.AddEmpty();
		connection.idx = i;
		connection.score = (float)scene.images[i].neighbors.GetSize();
		nPointsEstimate += ROUND2INT(depthData.depthMap.area()*(0.5f/*valid*/*0.3f/*new*/));
		if (depthData.normalMap.empty())
			bNormalMap = false;
	}
	connections.Sort();

	// fuse all depth-maps, processing the best connected images first
	const unsigned nMinViewsFuse(MINF(OPTDENSE::nMinViewsFuse, scene.images.GetSize()));
	const float normalError(COS(FD2R(OPTDENSE::fNormalDiffThreshold)));
	CLISTDEF0(Depth*) invalidDepths(0, 32);
	size_t nDepths(0);
	typedef TImage<cuint32_t> DepthIndex;
	typedef cList<DepthIndex> DepthIndexArr;
	DepthIndexArr arrDepthIdx(scene.images.GetSize());
	ProjsArr projs(0, nPointsEstimate);
	bool bEstimateNormal = OPTDENSE::nEstimateNormals;
	bool bEstimateColor = OPTDENSE::nEstimateColors;
	PointCloud pointcloud;
	pointcloud.Release();
	if (bEstimateNormal && !bNormalMap)
		bEstimateNormal = false;
	pointcloud.points.Reserve(nPointsEstimate);
	pointcloud.pointViews.Reserve(nPointsEstimate);
	pointcloud.pointWeights.Reserve(nPointsEstimate);
	if (bEstimateColor)
		pointcloud.colors.Reserve(nPointsEstimate);
	if (bEstimateNormal)
		pointcloud.normals.Reserve(nPointsEstimate);
	Util::Progress progress(_T("Fused depth-maps"), connections.GetSize());
	GET_LOGCONSOLE().Pause();
	FOREACHPTR(pConnection, connections) {
		TD_TIMER_STARTD();
		const uint32_t idxImage(pConnection->idx);
		const DepthData& depthData(arrDepthData[idxImage]);
		ASSERT(!depthData.images.IsEmpty() && !depthData.neighbors.IsEmpty());
		for (const ViewScore& neighbor: depthData.neighbors) {
			DepthIndex& depthIdxs = arrDepthIdx[neighbor.idx.ID];
			if (!depthIdxs.empty())
				continue;
			const DepthData& depthDataB(arrDepthData[neighbor.idx.ID]);
			if (depthDataB.IsEmpty())
				continue;
			depthIdxs.create(depthDataB.depthMap.size());
			depthIdxs.memset((uint8_t)NO_ID);
		}
		ASSERT(!depthData.IsEmpty());
		const Image8U::Size sizeMap(depthData.depthMap.size());
		const Image& imageData = *depthData.images.First().pImageData;
		ASSERT(&imageData-scene.images.Begin() == idxImage);
		DepthIndex& depthIdxs = arrDepthIdx[idxImage];
		if (depthIdxs.empty()) {
			depthIdxs.create(Image8U::Size(imageData.width, imageData.height));
			depthIdxs.memset((uint8_t)NO_ID);
		}
		const size_t nNumPointsPrev(pointcloud.points.GetSize());
		for (int i=0; i<sizeMap.height; ++i) {
			for (int j=0; j<sizeMap.width; ++j) {
				const ImageRef x(j,i);
				const Depth depth(depthData.depthMap(x));
				if (depth == 0)
					continue;
				++nDepths;
				ASSERT(ISINSIDE(depth, depthData.dMin, depthData.dMax));
				uint32_t& idxPoint = depthIdxs(x);
				if (idxPoint != NO_ID)
					continue;
				// create the corresponding 3D point
				idxPoint = (uint32_t)pointcloud.points.GetSize();
				PointCloud::Point& point = pointcloud.points.AddEmpty();
				point = imageData.camera.TransformPointI2W(Point3(Point2f(x),depth));
				PointCloud::ViewArr& views = pointcloud.pointViews.AddEmpty();
				views.Insert(idxImage);
				PointCloud::WeightArr& weights = pointcloud.pointWeights.AddEmpty();
				REAL confidence(weights.emplace_back(Conf2Weight(depthData.confMap(x),depth)));
				ProjArr& pointProjs = projs.AddEmpty();
				pointProjs.Insert(Proj(x));
				const PointCloud::Normal normal(bNormalMap ? Cast<Normal::Type>(imageData.camera.R.t()*Cast<REAL>(depthData.normalMap(x))) : Normal(0,0,-1));
				ASSERT(ISEQUAL(norm(normal), 1.f));
				// check the projection in the neighbor depth-maps
				Point3 X(point*confidence);
				Pixel32F C(Cast<float>(imageData.image(x))*confidence);
				PointCloud::Normal N(normal*confidence);
				invalidDepths.Empty();
				FOREACHPTR(pNeighbor, depthData.neighbors) {
					const IIndex idxImageB(pNeighbor->idx.ID);
					DepthData& depthDataB = arrDepthData[idxImageB];
					if (depthDataB.IsEmpty())
						continue;
					const Image& imageDataB = scene.images[idxImageB];
					const Point3f pt(imageDataB.camera.ProjectPointP3(point));
					if (pt.z <= 0)
						continue;
					const ImageRef xB(ROUND2INT(pt.x/pt.z), ROUND2INT(pt.y/pt.z));
					DepthMap& depthMapB = depthDataB.depthMap;
					if (!depthMapB.isInside(xB))
						continue;
					Depth& depthB = depthMapB(xB);
					if (depthB == 0)
						continue;
					uint32_t& idxPointB = arrDepthIdx[idxImageB](xB);
					if (idxPointB != NO_ID)
						continue;
					if (IsDepthSimilar(pt.z, depthB, OPTDENSE::fDepthDiffThreshold)) {
						// check if normals agree
						const PointCloud::Normal normalB(bNormalMap ? Cast<Normal::Type>(imageDataB.camera.R.t()*Cast<REAL>(depthDataB.normalMap(xB))) : Normal(0,0,-1));
						ASSERT(ISEQUAL(norm(normalB), 1.f));
						if (normal.dot(normalB) > normalError) {
							// add view to the 3D point
							ASSERT(views.FindFirst(idxImageB) == PointCloud::ViewArr::NO_INDEX);
							const float confidenceB(Conf2Weight(depthDataB.confMap(xB),depthB));
							const IIndex idx(views.InsertSort(idxImageB));
							weights.InsertAt(idx, confidenceB);
							pointProjs.InsertAt(idx, Proj(xB));
							idxPointB = idxPoint;
							X += imageDataB.camera.TransformPointI2W(Point3(Point2f(xB),depthB))*REAL(confidenceB);
							if (bEstimateColor)
								C += Cast<float>(imageDataB.image(xB))*confidenceB;
							if (bEstimateNormal)
								N += normalB*confidenceB;
							confidence += confidenceB;
							continue;
						}
					}
					if (pt.z < depthB) {
						// discard depth
						invalidDepths.Insert(&depthB);
					}
				}
				if (views.GetSize() < nMinViewsFuse) {
					// remove point
					FOREACH(v, views) {
						const IIndex idxImageB(views[v]);
						const ImageRef x(pointProjs[v].GetCoord());
						ASSERT(arrDepthIdx[idxImageB].isInside(x) && arrDepthIdx[idxImageB](x).idx != NO_ID);
						arrDepthIdx[idxImageB](x).idx = NO_ID;
					}
					projs.RemoveLast();
					pointcloud.pointWeights.RemoveLast();
					pointcloud.pointViews.RemoveLast();
					pointcloud.points.RemoveLast();
				} else {
					// this point is valid, store it
					const REAL nrm(REAL(1)/confidence);
					point = X*nrm;
					ASSERT(ISFINITE(point));
					if (bEstimateColor)
						pointcloud.colors.AddConstruct((C*(float)nrm).cast<uint8_t>());
					if (bEstimateNormal)
						pointcloud.normals.AddConstruct(normalized(N*(float)nrm));
					// invalidate all neighbor depths that do not agree with it
					for (Depth* pDepth: invalidDepths)
						*pDepth = 0;
				}
			}
		}
	}
    
	
	if(!connections.empty())
	{
		
		// FOREACHPTR(pConnection, connections) 
		// {
              const uint32_t idxImage(depthData.images.First().pImageData->ID); 
			  DepthData& depthData(arrDepthData[idxImage]);
			  const Image8U::Size size(depthData.images.First().image.size());
			  DepthIndex& depthIdxs = arrDepthIdx[idxImage]; 
			//   DepthData depthMap_fuse;
			  depthData.depthMap_fuse.create(size);
			//   NormalMap normalMap_fuse;
			  depthData.normalMap_fuse.create(size); 
              for (int i=0; i<size.height; ++i) 
			     for (int j=0; j<size.width; ++j) 
                     {
						ImageRef x(j,i);
						if(depthIdxs(x).idx == NO_ID )
						    {
								depthData.depthMap_fuse(x) = 0;
							    depthData.normalMap_fuse(x) = Normal::ZERO;
							}
						else
						  {
								depthData.depthMap_fuse(x) = depthData.depthMap(x);
								depthData.normalMap_fuse(x) = depthData.normalMap(x);
						  } 
					 }
                   ExportDepthMapByJetColormap(ComposeDepthInFusePath(depthData.GetView().GetID(), "png"), depthData.depthMap_fuse); 
				  // ExportNormalMap(ComposeNormalInFusePath(depthData.GetView().GetID(), "png"), depthData.normalMap_fuse); 
                //   ExportDepthMapByJetColormap(ComposeDepthInFusePath(depthData.GetView().GetID(), "prefuse.png"), depthData.depthMap); 
				//   ExportNormalMap(ComposeNormalInFusePath(depthData.GetView().GetID(), "prefuse.png"), depthData.normalMap); 
				//   depthMap_fuse.copyTo(depthData.depthMap_fuse);
				//   normalMap_fuse.copyTo(depthData.normalMap_fuse);
				//   ExportDepthMapByJetColormap(ComposeDepthInFusePath(depthData.GetView().GetID(), "aftercopy.png"), depthData.depthMap_fuse); 
				//   ExportNormalMap(ComposeNormalInFusePath(depthData.GetView().GetID(), "aftercopy.png"), depthData.normalMap_fuse); 

		// }


	}
	#endif
	return true;
} // RemoveSmallSegments
/*----------------------------------------------------------------*/

// try to fill small gaps in the depth map
bool DepthMapsData::GapInterpolation( DepthData& depthData)
{
	// std::cout<<"33333333333"<<std::endl;
	const float fDepthDiffThreshold(OPTDENSE::fDepthDiffThreshold*2.5f);
	unsigned nIpolGapSize = OPTDENSE::nIpolGapSize;
	DepthMap& depthMap = depthData.depthMap;
	NormalMap& normalMap = depthData.normalMap;
	ConfidenceMap& confMap = depthData.confMap;
	DepthMap& depthMap_fuse = depthData.depthMap_fuse;
	NormalMap& normalMap_fuse = depthData.normalMap_fuse;
	ASSERT(!depthMap.empty());
	const ImageRef size(depthMap.size());
	candidate.Empty();
	// 1. Row-wise:
	// for each row do
	for (int v=0; v<size.y; ++v) {
		// init counter
		unsigned count = 0;

		// for each element of the row do
		for (int u=0; u<size.x; ++u) {
			// get depth of this location
			const Depth& depth = depthMap_fuse(v,u);

			// if depth not valid => count and skip it
			if (depth <= 0) {
				++count;
				continue;
			}
			if (count == 0)
				continue;

			// check if speckle is small enough
			// and value in range
			
			if (count <= nIpolGapSize && (unsigned)u > count) {
				// first value index for interpolation
				int u_curr(u-count);
				const int u_first(u_curr-1);
				// compute mean depth
				const Depth& depthFirst = depthMap_fuse(v,u_first);
				if (IsDepthSimilar(depthFirst, depth, fDepthDiffThreshold)) {
					#if 0
					// set all values with the average
					const Depth avg((depthFirst+depth)*0.5f);
					do {
						depthMap(v,u_curr) = avg;
					} while (++u_curr<u);						
					#else
					// interpolate values
					const Depth diff((depth-depthFirst)/(count+1));
					Depth d(depthFirst);
					const float c(confMap.empty() ? 0.f : MINF(confMap(v,u_first), confMap(v,u)));
					if (normalMap_fuse.empty()) {
						do {
							depthMap_fuse(v,u_curr) = (d+=diff);
							if (!confMap.empty()) confMap(v,u_curr) = c;
						} while (++u_curr<u);						
					} else {
						Point2f dir1, dir2;
						Normal2Dir(normalMap_fuse(v,u_first), dir1);
						Normal2Dir(normalMap_fuse(v,u), dir2);
						const Point2f dirDiff((dir2-dir1)/float(count+1));
						do {
							depthMap_fuse(v,u_curr) = (d+=diff);
							dir1 += dirDiff;
							Dir2Normal(dir1, normalMap_fuse(v,u_curr));
							if (!confMap.empty()) confMap(v,u_curr) = c;
						} while (++u_curr<u);						
					}
					#endif
				}
			}
			
			if(count > nIpolGapSize && (unsigned)u > count){
				if((u-count) == 0){
					int u_next(u+1);
					const Depth& depthNext = depthMap_fuse(v,u_next);
					const Depth diff(depthNext-depth);
					for(int i=u ; i>=0 ;i--){
						ImageRef x0(i,v);
						ImageRef x1(u,v);
						float texture0(depthData.graMap(x0));
						float texture1(depthData.graMap(x1));
						
						float texture_difftmp = texture1-texture0;
						float texture_ratio = texture_difftmp/texture0;
						if(texture_ratio <= 0.1){
							Point2f dir1, dir2;
							Depth d(depth);
							Normal2Dir(normalMap_fuse(v,u), dir1);
							Normal2Dir(normalMap_fuse(v,u+1), dir2);
							const Point2f dirDiff(dir2-dir1);
							depthMap_fuse(v,i) = (d-=diff);
							dir1 -= dirDiff;
							Dir2Normal(dir1, normalMap_fuse(v,i));
						}		
					}			
				}
				if((u != size.x) && ((u-count) != 0)){
					int u_curr(u-count);
					const int u_first(u_curr-1);
					const Depth& depthFirst = depthMap_fuse(v,u_first);
					ImageRef x0(u_first,v);
					ImageRef x1(u,v);
					float texture0(depthData.graMap(x0));
					float texture1(depthData.graMap(x1));
					
					float texture_difftmp = texture1-texture0;
					float texture_ratio = texture_difftmp/texture0;
					if(texture_ratio <= 0.1 || IsDepthSimilar(depthFirst, depth, fDepthDiffThreshold)){
						#if 0
						// set all values with the average
						const Depth avg((depthFirst+depth)*0.5f);
						do {
							depthMap(v,u_curr) = avg;
						} while (++u_curr<u);						
						#else
						// interpolate values
						const Depth diff((depth-depthFirst)/(count+1));
						Depth d(depthFirst);
						const float c(confMap.empty() ? 0.f : MINF(confMap(v,u_first), confMap(v,u)));
						if (normalMap_fuse.empty()) {
							do {
								depthMap_fuse(v,u_curr) = (d+=diff);
								if (!confMap.empty()) confMap(v,u_curr) = c;
							} while (++u_curr<u);						
						} else {
							Point2f dir1, dir2;
							Normal2Dir(normalMap_fuse(v,u_first), dir1);
							Normal2Dir(normalMap_fuse(v,u), dir2);
							const Point2f dirDiff((dir2-dir1)/float(count+1));
							do {
								depthMap_fuse(v,u_curr) = (d+=diff);
								dir1 += dirDiff;
								Dir2Normal(dir1, normalMap_fuse(v,u_curr));
								if (!confMap.empty()) confMap(v,u_curr) = c;
							} while (++u_curr<u);						
						}
						#endif
						
					}
				}
				if(u == size.x){
					int u_curr(u-count);
					const int u_first(u_curr-1);
					int u_last(u_first-1);
					// compute mean depth
					const Depth& depthFirst = depthMap_fuse(v,u_first);
					const Depth& depthLast = depthMap_fuse(v,u_last);
					const Depth diff(depthLast-depthFirst);
					for(int i=1 ; i>=count ;i++){
						ImageRef x0(u_first,v);
						ImageRef x1(u_first+i,v);
						float texture0(depthData.graMap(x0));
						float texture1(depthData.graMap(x1));
						
						float texture_difftmp = texture1-texture0;
						float texture_ratio = texture_difftmp/texture0;
						if(texture_ratio <= 0.1){
							Point2f dir1, dir2;
							Depth d(depth);
							Normal2Dir(normalMap_fuse(v,u_first), dir1);
							Normal2Dir(normalMap_fuse(v,u_last), dir2);
							const Point2f dirDiff(dir2-dir1);
							depthMap_fuse(v,u_first+i) = (d-=diff);
							dir1 -= dirDiff;
							Dir2Normal(dir1, normalMap_fuse(v,u_first+i));
						}		
					}			
				}
				
			}
			
			// reset counter
			count = 0;
		}
	}

	// for (int v=size.y; v>=0; v--) {
	// 	// init counter
	// 	unsigned count = 0;

	// 	// for each element of the row do
	// 	for (int u=size.x; u>=0; u--) {
	// 		// get depth of this location
	// 		const Depth& depth = depthMap_fuse(v,u);

	// 		// if depth not valid => count and skip it
	// 		if (depth <= 0) {
	// 			++count;
	// 			continue;
	// 		}
	// 		if (count == 0)
	// 			continue;

	// 		// check if speckle is small enough
	// 		// and value in range
	// 		//小空洞补全
	// 		if (count <= nIpolGapSize && (unsigned)u > count) {
	// 			// first value index for interpolation
	// 			int u_curr(u+count);
	// 			const int u_first(u_curr+1);
	// 			// compute mean depth
	// 			const Depth& depthFirst = depthMap_fuse(v,u_first);
	// 			if (IsDepthSimilar(depthFirst, depth, fDepthDiffThreshold)) {
	// 				#if 0
	// 				// set all values with the average
	// 				const Depth avg((depthFirst+depth)*0.5f);
	// 				do {
	// 					depthMap(v,u_curr) = avg;
	// 				} while (++u_curr<u);						
	// 				#else
	// 				// interpolate values
	// 				const Depth diff((depth-depthFirst)/(count+1));
	// 				Depth d(depthFirst);
	// 				const float c(confMap.empty() ? 0.f : MINF(confMap(v,u_first), confMap(v,u)));
	// 				if (normalMap_fuse.empty()) {
	// 					do {
	// 						depthMap_fuse(v,u_curr) = (d+=diff);
	// 						if (!confMap.empty()) confMap(v,u_curr) = c;
	// 					} while (++u_curr<u);						
	// 				} else {
	// 					Point2f dir1, dir2;
	// 					Normal2Dir(normalMap_fuse(v,u_first), dir1);
	// 					Normal2Dir(normalMap_fuse(v,u), dir2);
	// 					const Point2f dirDiff((dir2-dir1)/float(count+1));
	// 					do {
	// 						depthMap_fuse(v,u_curr) = (d+=diff);
	// 						dir1 += dirDiff;
	// 						Dir2Normal(dir1, normalMap_fuse(v,u_curr));
	// 						if (!confMap.empty()) confMap(v,u_curr) = c;
	// 					} while (++u_curr<u);						
	// 				}
	// 				#endif
	// 			}
	// 		}
	// 		
	// 		if(count > nIpolGapSize && (unsigned)u > count){
	// 			if((u+count) == size.x){
	// 				int u_next(u-1);
	// 				const Depth& depthNext = depthMap_fuse(v,u_next);
	// 				const Depth diff(depthNext-depth);
	// 				for(int i=u+1 ; i<size.x ;i++){
	// 					ImageRef x0(i,v);
	// 					ImageRef x1(u,v);
	// 					float texture0(depthData.graMap(x0));
	// 					float texture1(depthData.graMap(x1));
	// 					
	// 					float texture_difftmp = texture1-texture0;
	// 					float texture_ratio = texture_difftmp/texture0;
	// 					if(texture_ratio <= 0.1){
	// 						Point2f dir1, dir2;
	// 						Depth d(depth);
	// 						Normal2Dir(normalMap_fuse(v,u), dir1);
	// 						Normal2Dir(normalMap_fuse(v,u-1), dir2);
	// 						const Point2f dirDiff(dir2-dir1);
	// 						depthMap_fuse(v,i) = (d-=diff);
	// 						dir1 -= dirDiff;
	// 						Dir2Normal(dir1, normalMap_fuse(v,i));
	// 					}		
	// 				}			
	// 			}
	// 			if((u != 0) && ((u+count) != size.x)){
	// 				int u_curr(u+count);
	// 				const int u_first(u_curr+1);
	// 				const Depth& depthFirst = depthMap_fuse(v,u_first);
	// 				ImageRef x0(u_first,v);
	// 				ImageRef x1(u,v);
	// 				float texture0(depthData.graMap(x0));
	// 				float texture1(depthData.graMap(x1));
	// 				
	// 				float texture_difftmp = texture1-texture0;
	// 				float texture_ratio = texture_difftmp/texture0;
	// 				if(texture_ratio <= 0.1 || IsDepthSimilar(depthFirst, depth, fDepthDiffThreshold)){
	// 					#if 0
	// 					// set all values with the average
	// 					const Depth avg((depthFirst+depth)*0.5f);
	// 					do {
	// 						depthMap(v,u_curr) = avg;
	// 					} while (++u_curr<u);						
	// 					#else
	// 					// interpolate values
	// 					const Depth diff((depth-depthFirst)/(count+1));
	// 					Depth d(depthFirst);
	// 					const float c(confMap.empty() ? 0.f : MINF(confMap(v,u_first), confMap(v,u)));
	// 					if (normalMap_fuse.empty()) {
	// 						do {
	// 							depthMap_fuse(v,u_curr) = (d+=diff);
	// 							if (!confMap.empty()) confMap(v,u_curr) = c;
	// 						} while (++u_curr<u);						
	// 					} else {
	// 						Point2f dir1, dir2;
	// 						Normal2Dir(normalMap_fuse(v,u_first), dir1);
	// 						Normal2Dir(normalMap_fuse(v,u), dir2);
	// 						const Point2f dirDiff((dir2-dir1)/float(count+1));
	// 						do {
	// 							depthMap_fuse(v,u_curr) = (d+=diff);
	// 							dir1 += dirDiff;
	// 							Dir2Normal(dir1, normalMap_fuse(v,u_curr));
	// 							if (!confMap.empty()) confMap(v,u_curr) = c;
	// 						} while (++u_curr<u);						
	// 					}
	// 					#endif
						
	// 				}
	// 			}
	// 			if(u == 0){
	// 				int u_curr(u+count);
	// 				const int u_first(u_curr+1);
	// 				int u_last(u_first+1);
	// 				// compute mean depth
	// 				const Depth& depthFirst = depthMap_fuse(v,u_first);
	// 				const Depth& depthLast = depthMap_fuse(v,u_last);
	// 				const Depth diff(depthLast-depthFirst);
	// 				for(int i=1 ; i>=u_curr ; i++){
	// 					ImageRef x0(u_first,v);
	// 					ImageRef x1(u_first-i,v);
	// 					float texture0(depthData.graMap(x0));
	// 					float texture1(depthData.graMap(x1));
	// 					
	// 					float texture_difftmp = texture1-texture0;
	// 					float texture_ratio = texture_difftmp/texture0;
	// 					if(texture_ratio <= 0.1){
	// 						Point2f dir1, dir2;
	// 						Depth d(depth);
	// 						Normal2Dir(normalMap_fuse(v,u_first), dir1);
	// 						Normal2Dir(normalMap_fuse(v,u_last), dir2);
	// 						const Point2f dirDiff(dir2-dir1);
	// 						depthMap_fuse(v,u_first-i) = (d-=diff);
	// 						dir1 -= dirDiff;
	// 						Dir2Normal(dir1, normalMap_fuse(v,u_first-i));
	// 					}		
	// 				}			
	// 			}
				
	// 		}
			
	// 		// reset counter
	// 		count = 0;
	// 	}
	// }

	// 2. Column-wise:
	// for each column do
	for (int u=0; u<size.x; ++u) {

		// init counter
		unsigned count = 0;

		// for each element of the column do
		for (int v=0; v<size.y; ++v) {
			// get depth of this location
			const Depth& depth = depthMap_fuse(v,u);

			// if depth not valid => count and skip it
			if (depth <= 0) {
				++count;
				continue;
			}
			if (count == 0)
				continue;

			// check if gap is small enough
			// and value in range
			if (count <= nIpolGapSize && (unsigned)v > count) {
				// first value index for interpolation
				int v_curr(v-count);
				const int v_first(v_curr-1);
				// compute mean depth
				const Depth& depthFirst = depthMap_fuse(v_first,u);
				if (IsDepthSimilar(depthFirst, depth, fDepthDiffThreshold)) {
					#if 0
					// set all values with the average
					const Depth avg((depthFirst+depth)*0.5f);
					do {
						depthMap(v_curr,u) = avg;
					} while (++v_curr<v);						
					#else
					// interpolate values
					const Depth diff((depth-depthFirst)/(count+1));
					Depth d(depthFirst);
					const float c(confMap.empty() ? 0.f : MINF(confMap(v_first,u), confMap(v,u)));
					if (normalMap_fuse.empty()) {
						do {
							depthMap_fuse(v_curr,u) = (d+=diff);
							if (!confMap.empty()) confMap(v_curr,u) = c;
						} while (++v_curr<v);						
					} else {
						Point2f dir1, dir2;
						Normal2Dir(normalMap_fuse(v_first,u), dir1);
						Normal2Dir(normalMap_fuse(v,u), dir2);
						const Point2f dirDiff((dir2-dir1)/float(count+1));
						do {
							depthMap_fuse(v_curr,u) = (d+=diff);
							dir1 += dirDiff;
							Dir2Normal(dir1, normalMap_fuse(v_curr,u));
							if (!confMap.empty()) confMap(v_curr,u) = c;
						} while (++v_curr<v);						
					}
					#endif
				}
			}
			
			if(count > nIpolGapSize && (unsigned)v > count){
				if((v-count) == 0){
					int v_next(v+1);
					const Depth& depthNext = depthMap_fuse(v_next,u);
					const Depth diff(depthNext-depth);
					for(int i=v ; i>=0 ;i--){
						ImageRef x0(u,i);
						ImageRef x1(u,v);
						float texture0(depthData.graMap(x0));
						float texture1(depthData.graMap(x1));
						
						float texture_difftmp = texture1-texture0;
						float texture_ratio = texture_difftmp/texture0;
						if(texture_ratio <= 0.1){
							Point2f dir1, dir2;
							Depth d(depth);
							Normal2Dir(normalMap_fuse(v,u), dir1);
							Normal2Dir(normalMap_fuse(v+1,u), dir2);
							const Point2f dirDiff(dir2-dir1);
							depthMap_fuse(i,u) = (d-=diff);
							dir1 -= dirDiff;
							Dir2Normal(dir1, normalMap_fuse(i,u));
						}		
					}			
				}
				if(((v-count) != 0) && (v != size.y)){
					int v_curr(v-count);
					const int v_first(v_curr-1);
					const Depth& depthFirst = depthMap_fuse(v_first,u);
					ImageRef x0(u,v_first);
					ImageRef x1(u,v);
					float texture0(depthData.graMap(x0));
					float texture1(depthData.graMap(x1));
					
					float texture_difftmp = texture1-texture0;
					float texture_ratio = texture_difftmp/texture0;
					if(texture_ratio <= 0.1 || IsDepthSimilar(depthFirst, depth, fDepthDiffThreshold)){
						#if 0
						// set all values with the average
						const Depth avg((depthFirst+depth)*0.5f);
						do {
							depthMap(v,u_curr) = avg;
						} while (++u_curr<u);						
						#else
						// interpolate values
						const Depth diff((depth-depthFirst)/(count+1));
						Depth d(depthFirst);
						const float c(confMap.empty() ? 0.f : MINF(confMap(v_first,u), confMap(v,u)));
						if (normalMap_fuse.empty()) {
							do {
								depthMap_fuse(v_curr,u) = (d+=diff);
								if (!confMap.empty()) confMap(v_curr,u) = c;
							} while (++v_curr<v);						
						} else {
							Point2f dir1, dir2;
							Normal2Dir(normalMap_fuse(v_first,u), dir1);
							Normal2Dir(normalMap_fuse(v,u), dir2);
							const Point2f dirDiff((dir2-dir1)/float(count+1));
							do {
								depthMap_fuse(v_curr,u) = (d+=diff);
								dir1 += dirDiff;
								Dir2Normal(dir1, normalMap_fuse(v_curr,u));
								if (!confMap.empty()) confMap(v_curr,u) = c;
							} while (++v_curr<v);						
						}
						#endif
					}
				}
				if(v == size.y){
					int v_curr(v-count);
					const int v_first(v_curr-1);
					int v_last(v_first-1);
					// compute mean depth
					const Depth& depthFirst = depthMap_fuse(v_first,u);
					const Depth& depthLast = depthMap_fuse(v_last,u);
					const Depth diff(depthLast-depthFirst);
					for(int i=1 ; i>=count ;i++){
						ImageRef x0(u,v_first);
						ImageRef x1(u,v_first+i);
						float texture0(depthData.graMap(x0));
						float texture1(depthData.graMap(x1));
						
						float texture_difftmp = texture1-texture0;
						float texture_ratio = texture_difftmp/texture0;
						if(texture_ratio <= 0.1){
							Point2f dir1, dir2;
							Depth d(depth);
							Normal2Dir(normalMap_fuse(v_first,u), dir1);
							Normal2Dir(normalMap_fuse(v_last,u), dir2);
							const Point2f dirDiff(dir2-dir1);
							depthMap_fuse(v_first+i,u) = (d-=diff);
							dir1 -= dirDiff;
							Dir2Normal(dir1, normalMap_fuse(v_first+i,u));
						}		
					}			
				}
			}
			
			// reset counter
			count = 0;
		}
	}
	// ExportDepthMapByJetColormap(ComposeDepthFileSegmentPath(depthData.GetView().GetID(), "1.png"), depthMap_fuse); 
	// ExportNormalMap(ComposeDepthFileSegmentPath(depthData.GetView().GetID(), "2.png"), normalMap_fuse);

	
	
	for (int v=0; v<size.y; ++v) {
		// for each element of the row do
		for (int u=0; u<size.x; ++u) {
			// get depth of this location
			Depth& depth_fuse = depthMap_fuse(v,u);
			// Depth& depth = depthMap(v,u);
			candidate.Empty();	

			if (depth_fuse <= 0) {
				ImageRef x0(u,v);
				float texture0(depthData.graMap(x0)); 
				int propagationHalfWindow = OPTDENSE::propagatehalfwin ;           
				int step = OPTDENSE::propagatestep ;						     
				if(texture0 > 150){  
					propagationHalfWindow = 5 ;			
				}
				else{
					propagationHalfWindow = OPTDENSE::propagatestep ;
				}
				if (x0.x > propagationHalfWindow && x0.y > propagationHalfWindow && x0.x < size.x-propagationHalfWindow && x0.y < size.y-propagationHalfWindow ){
					for (int i=1; i <= propagationHalfWindow; i+=step){
						candidate.push_back(Point2i(x0.x, x0.y - i)); 
						candidate.push_back(Point2i(x0.x + i, x0.y)); 
						candidate.push_back(Point2i(x0.x, x0.y + i)); 
						candidate.push_back(Point2i(x0.x - i, x0.y)); 
					}
				}
				
				
				float texture_diff = 0;
				float depth_diff = 0;
				Point2f x1_temax;
				Point2f x1_temin;
				Point2f x1_demax;
				Point2f x1_demin;
				Point2f dirDiff;
				Point2f dirDiffsum;
				float texture_ratiomin = 10 ;
				float texture_ratiomax = 0 ;
				float depth_ratiomin = 10 ;
				float depth_ratiomax = 0 ;
				float depthtmp = 0;
				float depthsum = 0;
				int count = 0;
				int dir_count = 0;
				int depth_count = 0;
				FOREACH(n, candidate){	
					const ImageRef& nx = candidate[n];
					float texture1 = depthData.graMap(nx);
					Depth ndepth(depthMap(nx));
					if(ndepth > 0){
						depth_count++;
						
						float texture_difftmp = texture1-texture0;
						texture_diff = texture_diff + texture_difftmp;
					
						float depth_difftmp = 0;
						if(n>0){
							depth_difftmp = ndepth-depthtmp;
							depth_diff = depth_diff + depth_difftmp;
							count++;
						}
						depthtmp = ndepth;
						depthsum +=ndepth;
						
						Point2f dir1, dir2;
						if(normalMap_fuse(nx) != Normal::ZERO){
							if(n>0){
								Normal2Dir(normalMap(nx), dir2);
								dirDiff = (dir2-dir1)/count;
								count = 0;
								dir_count++;
							}
							Normal2Dir(normalMap(nx), dir1);
							dirDiffsum +=dirDiff; 
						}
						
						
						float texture_ratiotmp = texture_difftmp / texture0;
						float depth_ratiotmp = depth_difftmp / (depthsum/candidate.GetSize());

						
						texture_ratiomax = (MAXF(texture_ratiomax,texture_ratiotmp));
						if(texture_ratiomax == texture_ratiotmp)
						{x1_temax = nx;}
						texture_ratiomin = (MINF(texture_ratiomin,texture_ratiotmp));
						if(texture_ratiomin == texture_ratiotmp)
						{x1_temin = nx;}

						depth_ratiomax = (MAXF(depth_ratiomax,depth_ratiotmp));
						if(depth_ratiomax == depth_ratiotmp)
						{x1_demax = nx;}
						depth_ratiomin = (MINF(depth_ratiomin,depth_ratiotmp));
						if(depth_ratiomin == depth_ratiotmp)
						{x1_demin = nx;}

					}

				}
				if(depth_count == 0)
					continue;
				
				float texture_ratio = texture_diff / depth_count*texture0;
				float depth_ratio = depth_diff / depthsum;
				if(texture_ratio<0.01){
					if(depth_ratio<0.01){
						const float c(confMap.empty() ? 0.f : MINF(confMap(x1_demin), confMap(x0)));
						Depth ndepth = depthMap_fuse(x1_demin);
						float Distance = SQRT(SQUARE(x0.x-x1_demin.x)+SQUARE(x0.y-x1_demin.y));
						if(normalMap_fuse(x0) == Normal::ZERO ){
							Point2f dirDiff_mean = dirDiffsum/dir_count;
							Point2f dir;
							Normal2Dir(normalMap_fuse(x1_demin), dir);
							dir = dirDiff_mean + dir;
							Dir2Normal(dir, normalMap_fuse(x0));
						}
						else{
							Point2f dir1, dir2;
							Normal2Dir(normalMap_fuse(x0), dir1);
							Normal2Dir(normalMap_fuse(x1_demin), dir2);
							const Point2f dirDiff((dir2-dir1)/(Distance));
							dir1 += dirDiff;
							Dir2Normal(dir1, normalMap_fuse(x0));
						}
						float depthmean = depthsum/depth_count;
						float diff = (ndepth - depthmean)/Distance;
						depthMap_fuse(x0) = (depthmean+=diff);
						if (!confMap.empty()) confMap(x0) = c;

					}
					else{
						const float c(confMap.empty() ? 0.f : MINF(confMap(x1_demin), confMap(x0)));
						float Distance0 = SQRT(SQUARE(x0.x-x1_demax.x)+SQUARE(x0.y-x1_demax.y));
						float Distance1 = SQRT(SQUARE(x1_demin.x-x1_demax.x)+SQUARE(x1_demin.y-x1_demax.y));
						float Distance2 = SQRT(SQUARE(x0.x-x1_demin.x)+SQUARE(x0.y-x1_demin.y));
						float diff = ((depthMap_fuse(x1_demin)-depthMap_fuse(x1_demax))/(Distance1));
						if(normalMap_fuse(x0) == Normal::ZERO ){
							normalMap_fuse(x0) = normalMap_fuse(x1_demin);
						}
						else{
							Point2f dir1, dir2;
							Normal2Dir(normalMap_fuse(x0), dir1);
							Normal2Dir(normalMap_fuse(x1_demin), dir2);
							const Point2f dirDiff((dir2-dir1)/(Distance2));
							dir1 += dirDiff;
							Dir2Normal(dir1, normalMap_fuse(x0));
						}	
						depthMap_fuse(x0) = diff*Distance0;
						if (!confMap.empty()) confMap(x0) = c;
					}
					
				}
				else{
					if(depth_ratiomin<0.01){
						const float c(confMap.empty() ? 0.f : MINF(confMap(x1_demin), confMap(x0)));
						depthMap_fuse(x0) = depthMap(x0);
						float Distance = SQRT(SQUARE(x0.x-x1_demin.x)+SQUARE(x0.y-x1_demin.y));
						if(normalMap_fuse(x0) == Normal::ZERO ){
							normalMap_fuse(x0) = normalMap_fuse(x1_demin);
						}
						else{
							Point2f dir1, dir2;
							Normal2Dir(normalMap_fuse(x0), dir1);
							Normal2Dir(normalMap_fuse(x1_demin), dir2);
							const Point2f dirDiff((dir2-dir1)/(Distance));
							dir1 += dirDiff;
							Dir2Normal(dir1, normalMap_fuse(x0));
						}
						if (!confMap.empty()) confMap(x0) = c;
					}
					if(texture_ratiomin<0.01){
						const float c(confMap.empty() ? 0.f : MINF(confMap(x1_temin), confMap(x0)));
						float diff = ((depthData.graMap(x1_temin)-texture0)/texture0);
						float Distance = SQRT(SQUARE(x0.x-x1_temin.x)+SQUARE(x0.y-x1_temin.y));
						if(normalMap_fuse(x0) == Normal::ZERO ){
							normalMap_fuse(x0) = normalMap_fuse(x1_temin);
						}
						else{
							Point2f dir1, dir2;
							Normal2Dir(normalMap(x0), dir1);
							Normal2Dir(normalMap(x1_demax), dir2);
							const Point2f dirDiff((dir2-dir1)/(Distance));
							depthMap_fuse(x0) = depthMap_fuse(x1_temin)*(1+diff);
							dir1 += dirDiff;
							Dir2Normal(dir1, normalMap_fuse(x0));
						}
						if (!confMap.empty()) confMap(x0) = c;
					}
				}
			}
		
		}
	}
	 ExportDepthMapByJetColormap(ComposeDepthFileSegmentPath(depthData.GetView().GetID(), "3.png"), depthMap_fuse); 
	 ExportNormalMap(ComposeDepthFileSegmentPath(depthData.GetView().GetID(), "4.png"), normalMap_fuse);
	
	
	for (int v=0; v<size.y; ++v) {
		for (int u=0; u<size.x; ++u) {	
			ImageRef x0(u,v);
			if(depthMap_fuse(x0) > 0){
				depthMap(x0) = depthMap_fuse(x0);
			}
			if(normalMap_fuse(x0) != Normal::ZERO){
				normalMap(x0) = normalMap_fuse(x0);
			}
		}
	}
	// ExportDepthMapByJetColormap(ComposeDepthFileSegmentPath(depthData.GetView().GetID(), "png"), depthMap);
	return true;
} // GapInterpolation
/*----------------------------------------------------------------*/


// filter depth-map, one pixel at a time, using confidence based fusion or neighbor pixels
bool DepthMapsData::FilterDepthMap(DepthData& depthDataRef, const IIndexArr& idxNeighbors, bool bAdjust)
{
	TD_TIMER_STARTD();

	// count valid neighbor depth-maps
	ASSERT(depthDataRef.IsValid() && !depthDataRef.IsEmpty());
	const IIndex N = idxNeighbors.GetSize();
	ASSERT(OPTDENSE::nMinViewsFilter > 0 && scene.nCalibratedImages > 1);
	const IIndex nMinViews(MINF(OPTDENSE::nMinViewsFilter,scene.nCalibratedImages-1));
	const IIndex nMinViewsAdjust(MINF(OPTDENSE::nMinViewsFilterAdjust,scene.nCalibratedImages-1));
	if (N < nMinViews || N < nMinViewsAdjust) {
		DEBUG("error: depth map %3u can not be filtered", depthDataRef.GetView().GetID());
		return false;
	}

	// project all neighbor depth-maps to this image
	const DepthData::ViewData& imageRef = depthDataRef.images.First();
	const Image8U::Size sizeRef(depthDataRef.depthMap.size());
	const Camera& cameraRef = imageRef.camera;
	DepthMapArr depthMaps(N);
	ConfidenceMapArr confMaps(N);
	FOREACH(n, depthMaps) {
		DepthMap& depthMap = depthMaps[n];
		depthMap.create(sizeRef);
		depthMap.memset(0);
		ConfidenceMap& confMap = confMaps[n];
		if (bAdjust) {
			confMap.create(sizeRef);
			confMap.memset(0);
		}
		const IIndex idxView = depthDataRef.neighbors[idxNeighbors[(IIndex)n]].idx.ID;
		const DepthData& depthData = arrDepthData[idxView];
		const Camera& camera = depthData.images.First().camera;
		const Image8U::Size size(depthData.depthMap.size());
		for (int i=0; i<size.height; ++i) {
			for (int j=0; j<size.width; ++j) {
				const ImageRef x(j,i);
				const Depth depth(depthData.depthMap(x));
				if (depth == 0)
					continue;
				ASSERT(depth > 0);
				const Point3 X(camera.TransformPointI2W(Point3(x.x,x.y,depth)));
				const Point3 camX(cameraRef.TransformPointW2C(X));
				if (camX.z <= 0)
					continue;
				#if 0
				// set depth on the rounded image projection only
				const ImageRef xRef(ROUND2INT(cameraRef.TransformPointC2I(camX)));
				if (!depthMap.isInside(xRef))
					continue;
				Depth& depthRef(depthMap(xRef));
				if (depthRef != 0 && depthRef < camX.z)
					continue;
				depthRef = camX.z;
				if (bAdjust)
					confMap(xRef) = depthData.confMap(x);
				#else
				// set depth on the 4 pixels around the image projection
				const Point2 imgX(cameraRef.TransformPointC2I(camX));
				const ImageRef xRefs[4] = {
					ImageRef(FLOOR2INT(imgX.x), FLOOR2INT(imgX.y)),
					ImageRef(FLOOR2INT(imgX.x), CEIL2INT(imgX.y)),
					ImageRef(CEIL2INT(imgX.x), FLOOR2INT(imgX.y)),
					ImageRef(CEIL2INT(imgX.x), CEIL2INT(imgX.y))
				};
				for (int p=0; p<4; ++p) {
					const ImageRef& xRef = xRefs[p];
					if (!depthMap.isInside(xRef))
						continue;
					Depth& depthRef(depthMap(xRef));
					if (depthRef != 0 && depthRef < (Depth)camX.z)
						continue;
					depthRef = (Depth)camX.z;
					if (bAdjust)
						confMap(xRef) = depthData.confMap(x);
				}
				#endif
			}
		}
		#if TD_VERBOSE != TD_VERBOSE_OFF
		if (g_nVerbosityLevel > 3)
			ExportDepthMap(MAKE_PATH(String::FormatString("depthRender%04u.%04u.png", depthDataRef.GetView().GetID(), idxView)), depthMap);
		#endif
	}

	const float thDepthDiff(OPTDENSE::fDepthDiffThreshold*1.2f);
	DepthMap newDepthMap(sizeRef);
	ConfidenceMap newConfMap(sizeRef);
	#if TD_VERBOSE != TD_VERBOSE_OFF
	size_t nProcessed(0), nDiscarded(0);
	#endif
	if (bAdjust) {
		// average similar depths, and decrease confidence if depths do not agree
		// (inspired by: "Real-Time Visibility-Based Fusion of Depth Maps", Merrell, 2007)
		for (int i=0; i<sizeRef.height; ++i) {
			for (int j=0; j<sizeRef.width; ++j) {
				const ImageRef xRef(j,i);
				const Depth depth(depthDataRef.depthMap(xRef));
				if (depth == 0) {
					newDepthMap(xRef) = 0;
					newConfMap(xRef) = 0;
					continue;
				}
				ASSERT(depth > 0);
				#if TD_VERBOSE != TD_VERBOSE_OFF
				++nProcessed;
				#endif
				// update best depth and confidence estimate with all estimates
				float posConf(depthDataRef.confMap(xRef)), negConf(0);
				Depth avgDepth(depth*posConf);
				unsigned nPosViews(0), nNegViews(0);
				unsigned n(N);
				do {
					const Depth d(depthMaps[--n](xRef));
					if (d == 0) {
						if (nPosViews + nNegViews + n < nMinViews)
							goto DiscardDepth;
						continue;
					}
					ASSERT(d > 0);
					// if (IsDepthSimilar(depth, d, thDepthDiff)) 
					if ( IsDepthSimilar(depth, d,  float(0.12)) )
					{
						// average similar depths
						const float c(confMaps[n](xRef));
						avgDepth += d*c;
						posConf += c;
						++nPosViews;
					} else {
						// penalize confidence
						if (depth > d) {
							// occlusion
							negConf += confMaps[n](xRef);
						} else {
							// free-space violation
							const DepthData& depthData = arrDepthData[depthDataRef.neighbors[idxNeighbors[n]].idx.ID];
							const Camera& camera = depthData.images.First().camera;
							const Point3 X(cameraRef.TransformPointI2W(Point3(xRef.x,xRef.y,depth)));
							const ImageRef x(ROUND2INT(camera.TransformPointW2I(X)));
							if (depthData.confMap.isInside(x)) {
								const float c(depthData.confMap(x));
								negConf += (c > 0 ? c : confMaps[n](xRef));
							} else
								negConf += confMaps[n](xRef);
						}
						++nNegViews;
					}
				} while (n);
				ASSERT(nPosViews+nNegViews >= nMinViews);
				// if enough good views and positive confidence...
				if (nPosViews >= nMinViewsAdjust && posConf > negConf && ISINSIDE(avgDepth/=posConf, depthDataRef.dMin, depthDataRef.dMax)) {
					// consider this pixel an inlier
					newDepthMap(xRef) = avgDepth;
					newConfMap(xRef) = posConf - negConf;
				} else {
					// consider this pixel an outlier
					DiscardDepth:
					newDepthMap(xRef) = 0;
					newConfMap(xRef) = 0;
					#if TD_VERBOSE != TD_VERBOSE_OFF
					++nDiscarded;
					#endif
				}
			}
		}
	} else {
		// remove depth if it does not agree with enough neighbors
		const float thDepthDiffStrict(OPTDENSE::fDepthDiffThreshold*0.8f);
		const unsigned nMinGoodViewsProc(75), nMinGoodViewsDeltaProc(65);
		const unsigned nDeltas(4);
		const unsigned nMinViewsDelta(nMinViews*(nDeltas-2));
		const ImageRef xDs[nDeltas] = { ImageRef(-1,0), ImageRef(1,0), ImageRef(0,-1), ImageRef(0,1) };
		for (int i=0; i<sizeRef.height; ++i) {
			for (int j=0; j<sizeRef.width; ++j) {
				const ImageRef xRef(j,i);
				const Depth depth(depthDataRef.depthMap(xRef));
				if (depth == 0) {
					newDepthMap(xRef) = 0;
					newConfMap(xRef) = 0;
					continue;
				}
				ASSERT(depth > 0);
				#if TD_VERBOSE != TD_VERBOSE_OFF
				++nProcessed;
				#endif
				// check if very similar with the neighbors projected to this pixel
				{
					unsigned nGoodViews(0);
					unsigned nViews(0);
					unsigned n(N);
					do {
						const Depth d(depthMaps[--n](xRef));
						if (d > 0) {
							// valid view
							++nViews;
							if (IsDepthSimilar(depth, d, thDepthDiffStrict)) {
								// agrees with this neighbor
								++nGoodViews;
							}
						}
					} while (n);
					if (nGoodViews < nMinViews || nGoodViews < nViews*nMinGoodViewsProc/100) {
						#if TD_VERBOSE != TD_VERBOSE_OFF
						++nDiscarded;
						#endif
						newDepthMap(xRef) = 0;
						newConfMap(xRef) = 0;
						continue;
					}
				}
				// check if similar with the neighbors projected around this pixel
				{
					unsigned nGoodViews(0);
					unsigned nViews(0);
					for (unsigned d=0; d<nDeltas; ++d) {
						const ImageRef xDRef(xRef+xDs[d]);
						unsigned n(N);
						do {
							const Depth d(depthMaps[--n](xDRef));
							if (d > 0) {
								// valid view
								++nViews;
								if (IsDepthSimilar(depth, d, thDepthDiff)) {
									// agrees with this neighbor
									++nGoodViews;
								}
							}
						} while (n);
					}
					if (nGoodViews < nMinViewsDelta || nGoodViews < nViews*nMinGoodViewsDeltaProc/100) {
						#if TD_VERBOSE != TD_VERBOSE_OFF
						++nDiscarded;
						#endif
						newDepthMap(xRef) = 0;
						newConfMap(xRef) = 0;
						continue;
					}
				}
				// enough good views, keep it
				newDepthMap(xRef) = depth;
				newConfMap(xRef) = depthDataRef.confMap(xRef);
			}
		}
	}
	if (!SaveDepthMap(ComposeDepthFilePath(imageRef.GetID(), "filtered.dmap"), newDepthMap) ||
		!SaveConfidenceMap(ComposeDepthFilePath(imageRef.GetID(), "filtered.cmap"), newConfMap))
		return false;

	#if TD_VERBOSE != TD_VERBOSE_OFF
	DEBUG("Depth map %3u filtered using %u other images: %u/%u depths discarded (%s)", imageRef.GetID(), N, nDiscarded, nProcessed, TD_TIMER_GET_FMT().c_str());
	#endif

	return true;
} // FilterDepthMap
/*----------------------------------------------------------------*/

// fuse all valid depth-maps in the same 3D point cloud;
// join points very likely to represent the same 3D point and
// filter out points blocking the view
void DepthMapsData::FuseDepthMaps(PointCloud& pointcloud, bool bEstimateColor, bool bEstimateNormal)
{
	TD_TIMER_STARTD();

	struct Proj {
		union {
			uint32_t idxPixel;
			struct {
				uint16_t x, y; // image pixel coordinates
			};
		};
		inline Proj() {}
		inline Proj(uint32_t _idxPixel) : idxPixel(_idxPixel) {}
		inline Proj(const ImageRef& ir) : x(ir.x), y(ir.y) {}
		inline ImageRef GetCoord() const { return ImageRef(x,y); }
	};
	typedef SEACAVE::cList<Proj,const Proj&,0,4,uint32_t> ProjArr;
	typedef SEACAVE::cList<ProjArr,const ProjArr&,1,65536> ProjsArr;

	// find best connected images
	IndexScoreArr connections(0, scene.images.GetSize());
	size_t nPointsEstimate(0);
	bool bNormalMap(true);
	FOREACH(i, scene.images) {
		DepthData& depthData = arrDepthData[i];
		if (!depthData.IsValid())
			continue;
		if (depthData.IncRef(ComposeDepthFilePath(depthData.GetView().GetID(), "dmap")) == 0)
			return;
		ASSERT(!depthData.IsEmpty());
		IndexScore& connection = connections.AddEmpty();
		connection.idx = i;
		connection.score = (float)scene.images[i].neighbors.GetSize();
		nPointsEstimate += ROUND2INT(depthData.depthMap.area()*(0.5f/*valid*/*0.3f/*new*/));
		if (depthData.normalMap.empty())
			bNormalMap = false;
	}
	connections.Sort();


	float  depthweight = OPTDENSE::depthweight  ;
	float  normalweight = OPTDENSE::normalweight  ;

	// fuse all depth-maps, processing the best connected images first
	const unsigned nMinViewsFuse(MINF(OPTDENSE::nMinViewsFuse, scene.images.GetSize()));
	const float normalError(COS(FD2R(OPTDENSE::fNormalDiffThreshold * normalweight)));			
	CLISTDEF0(Depth*) invalidDepths(0, 32);
	size_t nDepths(0);
	typedef TImage<cuint32_t> DepthIndex;
	typedef cList<DepthIndex> DepthIndexArr;
	DepthIndexArr arrDepthIdx(scene.images.GetSize());
	ProjsArr projs(0, nPointsEstimate);
	if (bEstimateNormal && !bNormalMap)
		bEstimateNormal = false;
	pointcloud.points.Reserve(nPointsEstimate);
	pointcloud.pointViews.Reserve(nPointsEstimate);
	pointcloud.pointWeights.Reserve(nPointsEstimate);
	if (bEstimateColor)
		pointcloud.colors.Reserve(nPointsEstimate);
	if (bEstimateNormal)
		pointcloud.normals.Reserve(nPointsEstimate);
	Util::Progress progress(_T("Fused depth-maps"), connections.GetSize());
	GET_LOGCONSOLE().Pause();
	FOREACHPTR(pConnection, connections) {
		TD_TIMER_STARTD();
		const uint32_t idxImage(pConnection->idx);
		const DepthData& depthData(arrDepthData[idxImage]);
		ASSERT(!depthData.images.IsEmpty() && !depthData.neighbors.IsEmpty());
		for (const ViewScore& neighbor: depthData.neighbors) {
			DepthIndex& depthIdxs = arrDepthIdx[neighbor.idx.ID];
			if (!depthIdxs.empty())
				continue;
			const DepthData& depthDataB(arrDepthData[neighbor.idx.ID]);
			if (depthDataB.IsEmpty())
				continue;
			depthIdxs.create(depthDataB.depthMap.size());
			depthIdxs.memset((uint8_t)NO_ID);
		}
		ASSERT(!depthData.IsEmpty());
		const Image8U::Size sizeMap(depthData.depthMap.size());
		const Image& imageData = *depthData.images.First().pImageData;
		ASSERT(&imageData-scene.images.Begin() == idxImage);
		DepthIndex& depthIdxs = arrDepthIdx[idxImage];
		if (depthIdxs.empty()) {
			depthIdxs.create(Image8U::Size(imageData.width, imageData.height));
			depthIdxs.memset((uint8_t)NO_ID);
		}
		const size_t nNumPointsPrev(pointcloud.points.GetSize());
		for (int i=0; i<sizeMap.height; ++i) {
			for (int j=0; j<sizeMap.width; ++j) {
				const ImageRef x(j,i);
				const Depth depth(depthData.depthMap(x));
				if (depth == 0)
					continue;
				++nDepths;
				ASSERT(ISINSIDE(depth, depthData.dMin, depthData.dMax));
				uint32_t& idxPoint = depthIdxs(x);
				if (idxPoint != NO_ID)
					continue;
				// create the corresponding 3D point
				idxPoint = (uint32_t)pointcloud.points.GetSize();
				PointCloud::Point& point = pointcloud.points.AddEmpty();
				point = imageData.camera.TransformPointI2W(Point3(Point2f(x),depth));
				PointCloud::ViewArr& views = pointcloud.pointViews.AddEmpty();
				views.Insert(idxImage);
				PointCloud::WeightArr& weights = pointcloud.pointWeights.AddEmpty();
				REAL confidence(weights.emplace_back(Conf2Weight(depthData.confMap(x),depth)));
				ProjArr& pointProjs = projs.AddEmpty();
				pointProjs.Insert(Proj(x));
				const PointCloud::Normal normal(bNormalMap ? Cast<Normal::Type>(imageData.camera.R.t()*Cast<REAL>(depthData.normalMap(x))) : Normal(0,0,-1));
				ASSERT(ISEQUAL(norm(normal), 1.f));
				// check the projection in the neighbor depth-maps
				Point3 X(point*confidence);
				Pixel32F C(Cast<float>(imageData.image(x))*confidence);
				PointCloud::Normal N(normal*confidence);
				invalidDepths.Empty();
				FOREACHPTR(pNeighbor, depthData.neighbors) {
					const IIndex idxImageB(pNeighbor->idx.ID);
					DepthData& depthDataB = arrDepthData[idxImageB];
					if (depthDataB.IsEmpty())
						continue;
					const Image& imageDataB = scene.images[idxImageB];
					const Point3f pt(imageDataB.camera.ProjectPointP3(point));
					if (pt.z <= 0)
						continue;
					const ImageRef xB(ROUND2INT(pt.x/pt.z), ROUND2INT(pt.y/pt.z));
					DepthMap& depthMapB = depthDataB.depthMap;
					if (!depthMapB.isInside(xB))
						continue;
					Depth& depthB = depthMapB(xB);
					if (depthB == 0)
						continue;
					uint32_t& idxPointB = arrDepthIdx[idxImageB](xB);
					if (idxPointB != NO_ID)
						continue;
					if (IsDepthSimilar(pt.z, depthB, OPTDENSE::fDepthDiffThreshold* depthweight )) {      
						// check if normals agree
						const PointCloud::Normal normalB(bNormalMap ? Cast<Normal::Type>(imageDataB.camera.R.t()*Cast<REAL>(depthDataB.normalMap(xB))) : Normal(0,0,-1));
						ASSERT(ISEQUAL(norm(normalB), 1.f));
						if (normal.dot(normalB) > normalError) {
							// add view to the 3D point
							ASSERT(views.FindFirst(idxImageB) == PointCloud::ViewArr::NO_INDEX);
							const float confidenceB(Conf2Weight(depthDataB.confMap(xB),depthB));
							const IIndex idx(views.InsertSort(idxImageB));
							weights.InsertAt(idx, confidenceB);
							pointProjs.InsertAt(idx, Proj(xB));
							idxPointB = idxPoint;
							X += imageDataB.camera.TransformPointI2W(Point3(Point2f(xB),depthB))*REAL(confidenceB);
							if (bEstimateColor)
								C += Cast<float>(imageDataB.image(xB))*confidenceB;
							if (bEstimateNormal)
								N += normalB*confidenceB;
							confidence += confidenceB;
							continue;
						}
					}
					if (pt.z < depthB) {
						// discard depth
						invalidDepths.Insert(&depthB);
					}
				}
				if (views.GetSize() < nMinViewsFuse) {
					// remove point
					FOREACH(v, views) {
						const IIndex idxImageB(views[v]);
						const ImageRef x(pointProjs[v].GetCoord());
						ASSERT(arrDepthIdx[idxImageB].isInside(x) && arrDepthIdx[idxImageB](x).idx != NO_ID);
						arrDepthIdx[idxImageB](x).idx = NO_ID;
					}
					projs.RemoveLast();
					pointcloud.pointWeights.RemoveLast();
					pointcloud.pointViews.RemoveLast();
					pointcloud.points.RemoveLast();
				} else {
					// this point is valid, store it
					const REAL nrm(REAL(1)/confidence);
					point = X*nrm;
					ASSERT(ISFINITE(point));
					if (bEstimateColor)
						pointcloud.colors.AddConstruct((C*(float)nrm).cast<uint8_t>());
					if (bEstimateNormal)
						pointcloud.normals.AddConstruct(normalized(N*(float)nrm));
					// invalidate all neighbor depths that do not agree with it
					for (Depth* pDepth: invalidDepths)
						*pDepth = 0;
				}
			}
		}
		ASSERT(pointcloud.points.GetSize() == pointcloud.pointViews.GetSize() && pointcloud.points.GetSize() == pointcloud.pointWeights.GetSize() && pointcloud.points.GetSize() == projs.GetSize());
		DEBUG_ULTIMATE("Depths map for reference image %3u fused using %u depths maps: %u new points (%s)", idxImage, depthData.images.GetSize()-1, pointcloud.points.GetSize()-nNumPointsPrev, TD_TIMER_GET_FMT().c_str());
		progress.display(pConnection-connections.Begin());
	}
	GET_LOGCONSOLE().Play();
	progress.close();
	arrDepthIdx.Release();

	DEBUG_EXTRA("Depth-maps fused and filtered: %u depth-maps, %u depths, %u points (%d%%%%) (%s)", connections.GetSize(), nDepths, pointcloud.points.GetSize(), ROUND2INT((100.f*pointcloud.points.GetSize())/nDepths), TD_TIMER_GET_FMT().c_str());

	if (bEstimateNormal && !pointcloud.points.IsEmpty() && pointcloud.normals.IsEmpty()) {
		// estimate normal also if requested (quite expensive if normal-maps not available)
		TD_TIMER_STARTD();
		pointcloud.normals.Resize(pointcloud.points.GetSize());
		const int64_t nPoints((int64_t)pointcloud.points.GetSize());
		#ifdef DENSE_USE_OPENMP
		#pragma omp parallel for
		#endif
		for (int64_t i=0; i<nPoints; ++i) {
			PointCloud::WeightArr& weights = pointcloud.pointWeights[i];
			ASSERT(!weights.IsEmpty());
			IIndex idxView(0);
			float bestWeight = weights.First();
			for (IIndex idx=1; idx<weights.GetSize(); ++idx) {
				const PointCloud::Weight& weight = weights[idx];
				if (bestWeight < weight) {
					bestWeight = weight;
					idxView = idx;
				}
			}
			const DepthData& depthData(arrDepthData[pointcloud.pointViews[i][idxView]]);
			ASSERT(depthData.IsValid() && !depthData.IsEmpty());
			depthData.GetNormal(projs[i][idxView].GetCoord(), pointcloud.normals[i]);
		}
		DEBUG_EXTRA("Normals estimated for the dense point-cloud: %u normals (%s)", pointcloud.points.GetSize(), TD_TIMER_GET_FMT().c_str());
	}

	// release all depth-maps
	FOREACHPTR(pDepthData, arrDepthData) {
		if (pDepthData->IsValid())
			pDepthData->DecRef();
	}
} // FuseDepthMaps
/*----------------------------------------------------------------*/



// S T R U C T S ///////////////////////////////////////////////////

DenseDepthMapData::DenseDepthMapData(Scene& _scene, int _nFusionMode)
	: scene(_scene), depthMaps(_scene), idxImage(0), sem(1), nFusionMode(_nFusionMode)
{
	if (nFusionMode < 0) {
		STEREO::SemiGlobalMatcher::CreateThreads(scene.nMaxThreads);
		if (nFusionMode == -1)
			OPTDENSE::nOptimize &= ~OPTDENSE::OPTIMIZE;
	}
}
DenseDepthMapData::~DenseDepthMapData()
{
	if (nFusionMode < 0)
		STEREO::SemiGlobalMatcher::DestroyThreads();
}

void DenseDepthMapData::SignalCompleteDepthmapFilter()
{
	ASSERT(idxImage > 0);
	if (Thread::safeDec(idxImage) == 0)
		sem.Signal((unsigned)images.GetSize()*2);
}
/*----------------------------------------------------------------*/



// S T R U C T S ///////////////////////////////////////////////////

static void* DenseReconstructionEstimateTmp(void*);
static void* DenseReconstructionFilterTmp(void*);

bool Scene::DenseReconstruction(int nFusionMode)
{
	DenseDepthMapData data(*this, nFusionMode);

	// estimate depth-maps
	if (!ComputeDepthMaps(data))
		return false;
	if (ABS(nFusionMode) == 1)
		return true;

	// fuse all depth-maps
	pointcloud.Release();
	data.depthMaps.FuseDepthMaps(pointcloud, OPTDENSE::nEstimateColors == 2, OPTDENSE::nEstimateNormals == 2);
	#if TD_VERBOSE != TD_VERBOSE_OFF
	if (g_nVerbosityLevel > 2) {
		// print number of points with 3+ views
		size_t nPoints1m(0), nPoints2(0), nPoints3p(0);
		FOREACHPTR(pViews, pointcloud.pointViews) {
			switch (pViews->GetSize())
			{
			case 0:
			case 1:
				++nPoints1m;
				break;
			case 2:
				++nPoints2;
				break;
			default:
				++nPoints3p;
			}
		}
		VERBOSE("Dense point-cloud composed of:\n\t%u points with 1- views\n\t%u points with 2 views\n\t%u points with 3+ views", nPoints1m, nPoints2, nPoints3p);
	}
	#endif

	if (!pointcloud.IsEmpty()) {
		if (pointcloud.colors.IsEmpty() && OPTDENSE::nEstimateColors == 1)
			EstimatePointColors(images, pointcloud);
		if (pointcloud.normals.IsEmpty() && OPTDENSE::nEstimateNormals == 1)
			EstimatePointNormals(images, pointcloud);
	}
	return true;
} // DenseReconstruction
/*----------------------------------------------------------------*/

// do first half of dense reconstruction: depth map computation
// results are saved to "data"
bool Scene::ComputeDepthMaps(DenseDepthMapData& data)
{
	{
	// maps global view indices to our list of views to be processed
	IIndexArr imagesMap;

	// prepare images for dense reconstruction (load if needed)
	{
		TD_TIMER_START();
		data.images.Reserve(images.GetSize());
		imagesMap.Resize(images.GetSize());
		#ifdef DENSE_USE_OPENMP
		bool bAbort(false);
		#pragma omp parallel for shared(data, bAbort)
		for (int_t ID=0; ID<(int_t)images.GetSize(); ++ID) {
			#pragma omp flush (bAbort)
			if (bAbort)
				continue;
			const IIndex idxImage((IIndex)ID);
		#else
		FOREACH(idxImage, images) {
		#endif
			// skip invalid, uncalibrated or discarded images
			Image& imageData = images[idxImage];
			if (!imageData.IsValid()) {
				#ifdef DENSE_USE_OPENMP
				#pragma omp critical
				#endif
				imagesMap[idxImage] = NO_ID;
				continue;
			}
			// map image index
			#ifdef DENSE_USE_OPENMP
			#pragma omp critical
			#endif
			{
				imagesMap[idxImage] = data.images.GetSize();
				data.images.Insert(idxImage);
			}
			// reload image at the appropriate resolution
			const unsigned nMaxResolution(imageData.RecomputeMaxResolution(OPTDENSE::nResolutionLevel, OPTDENSE::nMinResolution, OPTDENSE::nMaxResolution));
			if (!imageData.ReloadImage(nMaxResolution)) {
				#ifdef DENSE_USE_OPENMP
				bAbort = true;
				#pragma omp flush (bAbort)
				continue;
				#else
				return false;
				#endif
			}
			imageData.UpdateCamera(platforms);
			// print image camera
			DEBUG_ULTIMATE("K%d = \n%s", idxImage, cvMat2String(imageData.camera.K).c_str());
			DEBUG_LEVEL(3, "R%d = \n%s", idxImage, cvMat2String(imageData.camera.R).c_str());
			DEBUG_LEVEL(3, "C%d = \n%s", idxImage, cvMat2String(imageData.camera.C).c_str());
		}
		#ifdef DENSE_USE_OPENMP
		if (bAbort || data.images.IsEmpty()) {
		#else
		if (data.images.IsEmpty()) {
		#endif
			VERBOSE("error: preparing images for dense reconstruction failed (errors loading images)");
			return false;
		}
		VERBOSE("Preparing images for dense reconstruction completed: %d images (%s)", images.GetSize(), TD_TIMER_GET_FMT().c_str());
	}

	// select images to be used for dense reconstruction
	{
		TD_TIMER_START();
		// for each image, find all useful neighbor views
		IIndexArr invalidIDs;
		#ifdef DENSE_USE_OPENMP
		#pragma omp parallel for shared(data, invalidIDs)
		for (int_t ID=0; ID<(int_t)data.images.GetSize(); ++ID) {
			const IIndex idx((IIndex)ID);
		#else
		FOREACH(idx, data.images) {
		#endif
			const IIndex idxImage(data.images[idx]);
			ASSERT(imagesMap[idxImage] != NO_ID);
			DepthData& depthData(data.depthMaps.arrDepthData[idxImage]);
			if (!data.depthMaps.SelectViews(depthData)) {
				#ifdef DENSE_USE_OPENMP
				#pragma omp critical
				#endif
				invalidIDs.InsertSort(idx);
			}
		}
		RFOREACH(i, invalidIDs) {
			const IIndex idx(invalidIDs[i]);
			imagesMap[data.images.Last()] = idx;
			imagesMap[data.images[idx]] = NO_ID;
			data.images.RemoveAt(idx);
		}
		// globally select a target view for each reference image
		if (OPTDENSE::nNumViews == 1 && !data.depthMaps.SelectViews(data.images, imagesMap, data.neighborsMap)) {
			VERBOSE("error: no valid images to be dense reconstructed");
			return false;
		}
		ASSERT(!data.images.IsEmpty());
		VERBOSE("Selecting images for dense reconstruction completed: %d images (%s)", data.images.GetSize(), TD_TIMER_GET_FMT().c_str());
	}
	}

	for(int it = 0; it<OPTDENSE::nEstimationIters_external; it++)
	// for(int it = 0; it<4; it++)
	{
		
		data.it_external = it ;
		// initialize the queue of images to be processed
		data.idxImage = 0;
		ASSERT(data.events.IsEmpty());
		data.events.AddEvent(new EVTProcessImage(0));
		// start working threads
		data.progress = new Util::Progress("Estimated depth-maps", data.images.GetSize());
		GET_LOGCONSOLE().Pause();
		if (nMaxThreads > 1) 
		{
			// multi-thread execution
			cList<SEACAVE::Thread> threads(2);
			FOREACHPTR(pThread, threads)
				pThread->start(DenseReconstructionEstimateTmp, (void*)&data);
			FOREACHPTR(pThread, threads)
				pThread->join();
		} else 
		{
			// single-thread execution
			DenseReconstructionEstimate((void*)&data);
		}
		GET_LOGCONSOLE().Play();
		if (!data.events.IsEmpty())
			return false;
		data.progress.Release();
	}

	for(int i=0 ;i<data.depthMaps.arrDepthData.size();i++)
	{
		std::cout<<"depthData"<<i<<"isEmpty() = "<<data.depthMaps.arrDepthData[i].IsEmpty()<<std::endl ;
	}
	
	
	// if ((OPTDENSE::nOptimize & OPTDENSE::ADJUST_FILTER) != 0) 
	if(0)
	{
		// initialize the queue of depth-maps to be filtered
		data.sem.Clear();
		data.idxImage = data.images.GetSize();
		ASSERT(data.events.IsEmpty());
		FOREACH(i, data.images)
			data.events.AddEvent(new EVTFilterDepthMap(i));
		// start working threads
		data.progress = new Util::Progress("Filtered depth-maps", data.images.GetSize());
		GET_LOGCONSOLE().Pause();
		if (nMaxThreads > 1) {
			// multi-thread execution
			cList<SEACAVE::Thread> threads(MINF(nMaxThreads, (unsigned)data.images.GetSize()));
			FOREACHPTR(pThread, threads)
				pThread->start(DenseReconstructionFilterTmp, (void*)&data);
			FOREACHPTR(pThread, threads)
				pThread->join();
		} else {
			// single-thread execution
			DenseReconstructionFilter((void*)&data);
		}
		GET_LOGCONSOLE().Play();
		if (!data.events.IsEmpty())
			return false;
		data.progress.Release();

		
		FOREACH(idx, data.images)
		// for(int idx=0 ; idx<data.depthMaps.arrDepthData.size() ; idx++ )
		{
			DepthData& depthData(data.depthMaps.arrDepthData[data.images[idx]]);
			ExportDepthMapByJetColormap(ComposeInterFramePath(depthData.GetView().GetID(), "png"), depthData.depthMap); 
		}
	}

	// for(int i=0 ;i<data.depthMaps.arrDepthData.size();i++)
	// {
	// 	std::cout<<"depthData"<<i<<"isEmpty() = "<<data.depthMaps.arrDepthData[i].IsEmpty()<<std::endl ;
	// }

	
	// FOREACH(idx, data.images)
	// // for(int idx=0 ; idx<data.depthMaps.arrDepthData.size() ; idx++ )
	// {
	// 	DepthData& depthData(data.depthMaps.arrDepthData[data.images[idx]]);

	
	// 	std::cout<<"idx = "<<idx<<std::endl;
	// 	std::cout<<"depthData.GetView().GetID() = "<< depthData.GetView().GetID() <<std::endl;
		
	
	// 	if(depthData.IsEmpty())
	// 	{continue;}
		
	
    // 	data.depthMaps.LSC_superpixel(depthData);
	
	// 	ExportSuperPixelLabelsmap(ComposeSuperPixelLabelsPath(depthData.GetView().GetID(), "png"), depthData.superlabelMap); 


	
	// 	const Image8U::Size size(depthData.images.First().image.size());
	// 	depthData.superpriors_depthMap.create(size);
	// 	depthData.superpriors_normalMap.create(size);
	// 	for(int i=depthData.superpriors_depthMap.area() ; --i >= 0; ) 
	// 	{
	// 		depthData.superpriors_depthMap[i] = 0 ;
	// 		depthData.superpriors_normalMap[i] = Normal::ZERO;
	// 	}
	// 	const int block_num = 30 ;
	// 	int remainder = depthData.labelsNum % block_num ;
	// 	int integer = depthData.labelsNum / block_num ;
	// 	for(int i=0 ; i<integer ; i++)
	// 	{
	// 		data.depthMaps.GenerateSuperDepthPrior(depthData,i*block_num,block_num);
	// 	}

	// 	// ExportDepthMapByJetColormap(ComposeSuperPixelPriorsPath(depthData.GetView().GetID(), "png"), depthData.superpriors_depthMap); 
		
	// 	SaveDepthMap(ComposeSuperDepthPriorsPath(depthData.GetView().GetID(), "dmap"), depthData.superpriors_depthMap);
	// 	SaveNormalMap(ComposeSuperNormalPriorsPath(depthData.GetView().GetID(), "dmap"), depthData.superpriors_normalMap);

	// 	for(int i=depthData.superpriors_depthMap.area() ; --i >= 0; ) 
	// 	{
	// 		depthData.superpriors_depthMap[i] = 0 ;
	// 		depthData.superpriors_normalMap[i] = Normal::ZERO;
	// 	}

	// 	LoadDepthMap(ComposeSuperDepthPriorsPath(depthData.GetView().GetID(), "dmap"), depthData.superpriors_depthMap) ;
	// 	LoadNormalMap(ComposeSuperNormalPriorsPath(depthData.GetView().GetID(), "dmap"), depthData.superpriors_normalMap) ;
		

	// 	ExportDepthMapByJetColormap(ComposeSuperDepthPriorsPath(depthData.GetView().GetID(), "png"), depthData.superpriors_depthMap); 
	// 	ExportNormalMap(ComposeSuperNormalPriorsPath(depthData.GetView().GetID(), "normal.prior.png"), depthData.superpriors_normalMap);

	// }

	return true;
} // ComputeDepthMaps
/*----------------------------------------------------------------*/

void* DenseReconstructionEstimateTmp(void* arg) {
	const DenseDepthMapData& dataThreads = *((const DenseDepthMapData*)arg);
	dataThreads.scene.DenseReconstructionEstimate(arg);
	return NULL;
}

// initialize the dense reconstruction with the sparse point cloud
void Scene::DenseReconstructionEstimate(void* pData)
{
	DenseDepthMapData& data = *((DenseDepthMapData*)pData);
	while (true) {
		CAutoPtr<Event> evt(data.events.GetEvent());
		switch (evt->GetID()) {

		case EVT_PROCESSIMAGE: 
		{
			const EVTProcessImage& evtImage = *((EVTProcessImage*)(Event*)evt);
			if (evtImage.idxImage >= data.images.GetSize()) {
				if (nMaxThreads > 1) {
					// close working threads
					data.events.AddEvent(new EVTClose);
				}
				return;
			}

			const IIndex idx = data.images[evtImage.idxImage];
			DepthData& depthData(data.depthMaps.arrDepthData[idx]);

			// select views to reconstruct the depth-map for this image
			if(data.it_external == 0)
			{
				// init images pair: reference image and the best neighbor view
				ASSERT(data.neighborsMap.IsEmpty() || data.neighborsMap[evtImage.idxImage] != NO_ID);
				if (!data.depthMaps.InitViews(depthData, data.neighborsMap.IsEmpty()?NO_ID:data.neighborsMap[evtImage.idxImage], OPTDENSE::nNumViews)) 
				{
					// process next image
					data.events.AddEvent(new EVTProcessImage((IIndex)Thread::safeInc(data.idxImage)));
					break;
				}
			}
			// try to load already compute depth-map for this image
			if (data.nFusionMode >= 0 && File::access(ComposeDepthFilePath(depthData.GetView().GetID(), "dmap"))) 
			{
				if (OPTDENSE::nOptimize & OPTDENSE::OPTIMIZE) 
				{
					if (!depthData.Load(ComposeDepthFilePath(depthData.GetView().GetID(), "dmap"))) 
					{
						VERBOSE("error: invalid depth-map '%s'", ComposeDepthFilePath(depthData.GetView().GetID(), "dmap").c_str());
						exit(EXIT_FAILURE);
					}
					// optimize depth-map
					data.events.AddEventFirst(new EVTOptimizeDepthMap(evtImage.idxImage));
				}
				// process next image
				std::cout << " load   dmap   :     " <<  ComposeDepthFilePath(depthData.GetView().GetID(), "dmap" )<< std::endl ;
				data.events.AddEvent(new EVTProcessImage((uint32_t)Thread::safeInc(data.idxImage)));
			}
			else 
			{
				// estimate depth-map
				data.events.AddEventFirst(new EVTEstimateDepthMap(evtImage.idxImage));
			}
			break; 
		}

		case EVT_ESTIMATEDEPTHMAP: 
		{
			const EVTEstimateDepthMap& evtImage = *((EVTEstimateDepthMap*)(Event*)evt);
			// request next image initialization to be performed while computing this depth-map
			data.events.AddEvent(new EVTProcessImage((uint32_t)Thread::safeInc(data.idxImage)));
			// extract depth map
			data.sem.Wait();
			if (data.nFusionMode >= 0) {
				// extract depth-map using Patch-Match algorithm
				data.depthMaps.EstimateDepthMap(data.it_external, data.images[evtImage.idxImage]);
			} else {
				// extract disparity-maps using SGM algorithm
				if (data.nFusionMode == -1) {
					data.sgm.Match(*this, data.images[evtImage.idxImage], OPTDENSE::nNumViews);
				} else {
					// fuse existing disparity-maps
					const IIndex idx(data.images[evtImage.idxImage]);
					DepthData& depthData(data.depthMaps.arrDepthData[idx]);
					data.sgm.Fuse(*this, data.images[evtImage.idxImage], OPTDENSE::nNumViews, 2, depthData.depthMap, depthData.confMap);
					if (OPTDENSE::nEstimateNormals == 2)
						EstimateNormalMap(depthData.images.front().camera.K, depthData.depthMap, depthData.normalMap);
					depthData.dMin = ZEROTOLERANCE<float>(); depthData.dMax = FLT_MAX;
				}
			}
			data.sem.Signal();
			if(data.it_external == 1 || data.it_external == 2){
				// std::cout<<"1111111111"<<std::endl;
				if (OPTDENSE::nOptimize & OPTDENSE::OPTIMIZE) 
				{
					// optimize depth-map
					data.events.AddEventFirst(new EVTOptimizeDepthMap(evtImage.idxImage));
				} else {
					// save depth-map
					data.events.AddEventFirst(new EVTSaveDepthMap(evtImage.idxImage));
				}
				break; 
			}
			
		}

		case EVT_OPTIMIZEDEPTHMAP: {			
			const EVTOptimizeDepthMap& evtImage = *((EVTOptimizeDepthMap*)(Event*)evt);
			const IIndex idx = data.images[evtImage.idxImage];
			DepthData& depthData(data.depthMaps.arrDepthData[idx]);
			#if TD_VERBOSE != TD_VERBOSE_OFF
			// save depth map as image
			if (g_nVerbosityLevel > 3)
				ExportDepthMap(ComposeDepthFilePath(depthData.GetView().GetID(), "raw.png"), depthData.depthMap);
			#endif
			// apply filters
			if(data.it_external == 1 || data.it_external == 2){
					// if (OPTDENSE::nOptimize & (OPTDENSE::REMOVE_SPECKLES)) 
				{
					TD_TIMER_START();
					if (data.depthMaps.RemoveSmallSegments(depthData)) 
					{
						DEBUG_ULTIMATE("Depth-map %3u filtered: remove small segments (%s)", depthData.GetView().GetID(), TD_TIMER_GET_FMT().c_str());
					}
				}
				
				// if(data.it_external == 1)
				// if (OPTDENSE::nOptimize & (OPTDENSE::FILL_GAPS)) 
				{	
					TD_TIMER_START();
					if (data.depthMaps.GapInterpolation(depthData)) 
					{
						DEBUG_ULTIMATE("Depth-map %3u filtered: gap interpolation (%s)", depthData.GetView().GetID(), TD_TIMER_GET_FMT().c_str());
					}
				}
			}
			
			// ExportDepthMapByJetColormap(ComposeDepthFileSegmentPath(depthData.GetView().GetID(), "png"), depthData.depthMap); 

			// save depth-map
			data.events.AddEventFirst(new EVTSaveDepthMap(evtImage.idxImage));
			break; 
		}
		
		case EVT_SAVEDEPTHMAP: {
			const EVTSaveDepthMap& evtImage = *((EVTSaveDepthMap*)(Event*)evt);
			const IIndex idx = data.images[evtImage.idxImage];
			DepthData& depthData(data.depthMaps.arrDepthData[idx]);
			#if TD_VERBOSE != TD_VERBOSE_OFF
			// save depth map as image
			if (g_nVerbosityLevel > 2) {
				ExportDepthMap(ComposeDepthFilePath(depthData.GetView().GetID(), "png"), depthData.depthMap);
				ExportConfidenceMap(ComposeDepthFilePath(depthData.GetView().GetID(), "conf.png"), depthData.confMap);
				ExportPointCloud(ComposeDepthFilePath(depthData.GetView().GetID(), "ply"), *depthData.images.First().pImageData, depthData.depthMap, depthData.normalMap);
				if (g_nVerbosityLevel > 4) {
					ExportNormalMap(ComposeDepthFilePath(depthData.GetView().GetID(), "normal.png"), depthData.normalMap);
					depthData.confMap.Save(ComposeDepthFilePath(depthData.GetView().GetID(), "conf.pfm"));
				}
			}
			#endif
			// save compute depth-map for this image
			if(data.it_external == OPTDENSE::nEstimationIters_external-1)
			{
				if (!depthData.depthMap.empty())
					SaveDepthMap(ComposeResizeDepthFilePath(depthData.GetView().GetID(), "dmap"), depthData.depthMap);
					SaveNormalMap(ComposeResizeNormalFilePath(depthData.GetView().GetID(), "dmap"), depthData.normalMap);

				// if (!depthData.GetView().depthMapPrior.empty())
					// SaveDepthMap(ComposeDepthFilePath(depthData.GetView().GetID(), "prior.dmap"), depthData.GetView().depthMapPrior);
			}
			// depthData.ReleaseImages();
			// depthData.Release();
			data.progress->operator++();
			break; 
		}

		case EVT_CLOSE: {
			return; }

		default:
			ASSERT("Should not happen!" == NULL);
		}
	}
} // DenseReconstructionEstimate
/*----------------------------------------------------------------*/


void DepthMapsData::LSC_superpixel(DepthData& depthData)
{
	const DepthData::ViewData& image(depthData.images.First());
	const Image8U::Size size(image.image.size());
	depthData.superlabelMap.create(size);

    int region_size = 50;
    int min_element_size = 40;
    int num_iterations = 3;
	float depth_validpercent = 0.6;//0.6
    cv::Mat frame; 
    image.pImageData->image.copyTo(frame) ;

    cv::Mat result, mask;
    result = frame;
	cv::Mat converted;
	cvtColor(frame, converted, cv::COLOR_BGR2HSV); 

	cv::Ptr<cv::ximgproc::SuperpixelLSC>lsc = cv::ximgproc::createSuperpixelLSC(converted, region_size);
	lsc->iterate(num_iterations);
	if (min_element_size > 0)
		lsc->enforceLabelConnectivity(min_element_size);
	
	
	depthData.labelsNum = lsc->getNumberOfSuperpixels();
	

	// get the contours for displaying
	
	lsc->getLabelContourMask(mask, true);
	result.setTo(cv::Scalar(0, 0, 0), mask);   
	lsc->m_klabels2labels(depthData.superlabelMap); 

   
	// cv::imwrite( ComposeSuperPixelLabelsPath(depthData.GetView().GetID(), "png") , result );

	
	const int width = depthData.superlabelMap.size().width;
    const int height = depthData.superlabelMap.size().height;
	// std::cout<<"width = " << width <<std::endl  ;
    // std::cout<<"height = "<< height <<std::endl ;

	
	for(int i = 0; i < depthData.labelsNum; i++)
	{
		Point2i x(0,0) ;
		std::vector<Point2i> vector0(1,x) ;
		depthData.segmentLab.push_back(vector0);
	}
	for(int row = 0; row < height; ++row)
	{
        for (int col = 0; col < width; ++col) 
		{	
			Point2i x(col,row) ;
			int id = depthData.superlabelMap(x) ;
			depthData.segmentLab[id].push_back(x) ;
		}
	}

	for(int i=0 ; i<depthData.segmentLab.size() ; i++)
	{	
		int valid_depthNum = 0 ;
		for(int j=1 ; j<depthData.segmentLab[i].size() ; j++ )
		{
			const Point2i x = depthData.segmentLab[i][j] ;
			if(depthData.depthMap(x)!=0)
			{valid_depthNum++;}
		}

		if( valid_depthNum<depthData.segmentLab[i].size() * depth_validpercent)
		{
			depthData.segmentLab.erase(depthData.segmentLab.begin()+i);
		}
	}
	depthData.labelsNum = depthData.segmentLab.size() ;

	lsc.release();
	frame.release();
	mask.release();
	result.release();
}


void* DenseReconstructionFilterTmp(void* arg) {
	DenseDepthMapData& dataThreads = *((DenseDepthMapData*)arg);
	dataThreads.scene.DenseReconstructionFilter(arg);
	return NULL;
}

// filter estimated depth-maps
void Scene::DenseReconstructionFilter(void* pData)
{
	DenseDepthMapData& data = *((DenseDepthMapData*)pData);
	CAutoPtr<Event> evt;
	while ((evt=data.events.GetEvent(0)) != NULL) {
		switch (evt->GetID()) {
		case EVT_FILTERDEPTHMAP: {
			const EVTFilterDepthMap& evtImage = *((EVTFilterDepthMap*)(Event*)evt);
			const IIndex idx = data.images[evtImage.idxImage];
			DepthData& depthData(data.depthMaps.arrDepthData[idx]);
			if (!depthData.IsValid()) {
				data.SignalCompleteDepthmapFilter();
				break;
			}
			// make sure all depth-maps are loaded
			depthData.IncRef(ComposeDepthFilePath(depthData.GetView().GetID(), "dmap"));
			const unsigned numMaxNeighbors(8);
			IIndexArr idxNeighbors(0, depthData.neighbors.GetSize());
			FOREACH(n, depthData.neighbors) {
				const IIndex idxView = depthData.neighbors[n].idx.ID;
				DepthData& depthDataPair = data.depthMaps.arrDepthData[idxView];
				if (!depthDataPair.IsValid())
					continue;
				if (depthDataPair.IncRef(ComposeDepthFilePath(depthDataPair.GetView().GetID(), "dmap")) == 0) {
					// signal error and terminate
					data.events.AddEventFirst(new EVTFail);
					return;
				}
				idxNeighbors.Insert(n);
				if (idxNeighbors.GetSize() == numMaxNeighbors)
					break;
			}
			// filter the depth-map for this image
			if (data.depthMaps.FilterDepthMap(depthData, idxNeighbors, OPTDENSE::bFilterAdjust)) {
				// load the filtered maps after all depth-maps were filtered
				data.events.AddEvent(new EVTAdjustDepthMap(evtImage.idxImage));
			}
			// unload referenced depth-maps
			FOREACHPTR(pIdxNeighbor, idxNeighbors) {
				const IIndex idxView = depthData.neighbors[*pIdxNeighbor].idx.ID;
				DepthData& depthDataPair = data.depthMaps.arrDepthData[idxView];
				depthDataPair.DecRef();
			}
			depthData.DecRef();
			data.SignalCompleteDepthmapFilter();
			break; }

		case EVT_ADJUSTDEPTHMAP: {
			const EVTAdjustDepthMap& evtImage = *((EVTAdjustDepthMap*)(Event*)evt);
			const IIndex idx = data.images[evtImage.idxImage];
			DepthData& depthData(data.depthMaps.arrDepthData[idx]);
			ASSERT(depthData.IsValid());
			data.sem.Wait();
			// load filtered maps
			if (depthData.IncRef(ComposeDepthFilePath(depthData.GetView().GetID(), "dmap")) == 0 ||
				!LoadDepthMap(ComposeDepthFilePath(depthData.GetView().GetID(), "filtered.dmap"), depthData.depthMap) ||
				!LoadConfidenceMap(ComposeDepthFilePath(depthData.GetView().GetID(), "filtered.cmap"), depthData.confMap))
			{
				// signal error and terminate
				data.events.AddEventFirst(new EVTFail);
				return;
			}
			ASSERT(depthData.GetRef() == 1);
			File::deleteFile(ComposeDepthFilePath(depthData.GetView().GetID(), "filtered.dmap").c_str());
			File::deleteFile(ComposeDepthFilePath(depthData.GetView().GetID(), "filtered.cmap").c_str());
			#if TD_VERBOSE != TD_VERBOSE_OFF
			// save depth map as image
			if (g_nVerbosityLevel > 2) {
				ExportDepthMap(ComposeDepthFilePath(depthData.GetView().GetID(), "filtered.png"), depthData.depthMap);
				ExportPointCloud(ComposeDepthFilePath(depthData.GetView().GetID(), "filtered.ply"), *depthData.images.First().pImageData, depthData.depthMap, depthData.normalMap);
			}
			#endif
			// save filtered depth-map for this image
			depthData.Save(ComposeDepthFilePath(depthData.GetView().GetID(), "dmap"));
			data.progress->operator++();
			break; }

		case EVT_FAIL: {
			data.events.AddEventFirst(new EVTFail);
			return; }

		default:
			ASSERT("Should not happen!" == NULL);
		}
	}
} // DenseReconstructionFilter
/*----------------------------------------------------------------*/

// filter point-cloud based on camera-point visibility intersections
void Scene::PointCloudFilter(int thRemove)
{
	TD_TIMER_STARTD();

	typedef TOctree<PointCloud::PointArr,PointCloud::Point::Type,3,uint32_t,128> Octree;
	struct Collector {
		typedef Octree::IDX_TYPE IDX;
		typedef PointCloud::Point::Type Real;
		typedef TCone<Real,3> Cone;
		typedef TSphere<Real,3> Sphere;
		typedef TConeIntersect<Real,3> ConeIntersect;

		Cone cone;
		const ConeIntersect coneIntersect;
		const PointCloud& pointcloud;
		IntArr& visibility;
		PointCloud::Index idxPoint;
		Real distance;
		int weight;
		#ifdef DENSE_USE_OPENMP
		uint8_t pcs[sizeof(CriticalSection)];
		#endif

		Collector(const Cone::RAY& ray, Real angle, const PointCloud& _pointcloud, IntArr& _visibility)
			: cone(ray, angle), coneIntersect(cone), pointcloud(_pointcloud), visibility(_visibility)
		#ifdef DENSE_USE_OPENMP
		{ new(pcs) CriticalSection; }
		~Collector() { reinterpret_cast<CriticalSection*>(pcs)->~CriticalSection(); }
		inline CriticalSection& GetCS() { return *reinterpret_cast<CriticalSection*>(pcs); }
		#else
		{}
		#endif
		inline void Init(PointCloud::Index _idxPoint, const PointCloud::Point& X, int _weight) {
			const Real thMaxDepth(1.02f);
			idxPoint =_idxPoint;
			const PointCloud::Point::EVec D((PointCloud::Point::EVec&)X-cone.ray.m_pOrig);
			distance = D.norm();
			cone.ray.m_vDir = D/distance;
			cone.maxHeight = MaxDepthDifference(distance, thMaxDepth);
			weight = _weight;
		}
		inline bool Intersects(const Octree::POINT_TYPE& center, Octree::Type radius) const {
			return coneIntersect(Sphere(center, radius*Real(SQRT_3)));
		}
		inline void operator () (const IDX* idices, IDX size) {
			const Real thSimilar(0.01f);
			Real dist;
			FOREACHRAWPTR(pIdx, idices, size) {
				const PointCloud::Index idx(*pIdx);
				if (coneIntersect.Classify(pointcloud.points[idx], dist) == VISIBLE && !IsDepthSimilar(distance, dist, thSimilar)) {
					if (dist > distance)
						visibility[idx] += pointcloud.pointViews[idx].size();
					else
						visibility[idx] -= weight;
				}
			}
		}
	};
	typedef CLISTDEF2(Collector) Collectors;

	// create octree to speed-up search
	Octree octree(pointcloud.points);
	IntArr visibility(pointcloud.GetSize()); visibility.Memset(0);
	Collectors collectors; collectors.reserve(images.size());
	FOREACH(idxView, images) {
		const Image& image = images[idxView];
		const Ray3f ray(Cast<float>(image.camera.C), Cast<float>(image.camera.Direction()));
		const float angle(float(image.ComputeFOV(0)/image.width));
		collectors.emplace_back(ray, angle, pointcloud, visibility);
	}

	// run all camera-point visibility intersections
	Util::Progress progress(_T("Point visibility checks"), pointcloud.GetSize());
	#ifdef DENSE_USE_OPENMP
	#pragma omp parallel for //schedule(dynamic)
	for (int64_t i=0; i<(int64_t)pointcloud.GetSize(); ++i) {
		const PointCloud::Index idxPoint((PointCloud::Index)i);
	#else
	FOREACH(idxPoint, pointcloud.points) {
	#endif
		const PointCloud::Point& X = pointcloud.points[idxPoint];
		const PointCloud::ViewArr& views = pointcloud.pointViews[idxPoint];
		for (PointCloud::View idxView: views) {
			Collector& collector = collectors[idxView];
			#ifdef DENSE_USE_OPENMP
			Lock l(collector.GetCS());
			#endif
			collector.Init(idxPoint, X, (int)views.size());
			octree.Collect(collector, collector);
		}
		++progress;
	}
	progress.close();

	#if TD_VERBOSE != TD_VERBOSE_OFF
	if (g_nVerbosityLevel > 2) {
		// print visibility stats
		UnsignedArr counts(0, 64);
		for (int views: visibility) {
			if (views > 0)
				continue;
			while (counts.size() <= IDX(-views))
				counts.push_back(0);
			++counts[-views];
		}
		String msg;
		msg.reserve(64*counts.size());
		FOREACH(c, counts)
			if (counts[c])
				msg += String::FormatString("\n\t% 3u - % 9u", c, counts[c]);
		VERBOSE("Visibility lengths (%u points):%s", pointcloud.GetSize(), msg.c_str());
		// save outlier points
		PointCloud pc;
		RFOREACH(idxPoint, pointcloud.points) {
			if (visibility[idxPoint] <= thRemove) {
				pc.points.push_back(pointcloud.points[idxPoint]);
				pc.colors.push_back(pointcloud.colors[idxPoint]);
			}
		}
		pc.Save(MAKE_PATH("scene_dense_outliers.ply"));
	}
	#endif

	// filter points
	const size_t numInitPoints(pointcloud.GetSize());
	RFOREACH(idxPoint, pointcloud.points) {
		if (visibility[idxPoint] <= thRemove)
			pointcloud.RemovePoint(idxPoint);
	}

	DEBUG_EXTRA("Point-cloud filtered: %u/%u points (%d%%%%) (%s)", pointcloud.points.size(), numInitPoints, ROUND2INT((100.f*pointcloud.points.GetSize())/numInitPoints), TD_TIMER_GET_FMT().c_str());
} // PointCloudFilter
/*----------------------------------------------------------------*/
