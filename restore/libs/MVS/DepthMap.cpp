/*
* DepthMap.cpp
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
#include "DepthMap.h"
#define _USE_OPENCV
#include "Interface.h"
#include "../Common/AutoEstimator.h"
// CGAL: depth-map initialization
#include <CGAL/Simple_cartesian.h>
#include <CGAL/Delaunay_triangulation_2.h>
#include <CGAL/Projection_traits_xy_3.h>
// CGAL: estimate normals
#include <CGAL/Simple_cartesian.h>
#include <CGAL/property_map.h>
#include <CGAL/pca_estimate_normals.h>
using namespace MVS;


// D E F I N E S ///////////////////////////////////////////////////

#define DEFVAR_OPTDENSE_string(name, title, desc, ...)  DEFVAR_string(OPTDENSE, name, title, desc, __VA_ARGS__)
#define DEFVAR_OPTDENSE_bool(name, title, desc, ...)    DEFVAR_bool(OPTDENSE, name, title, desc, __VA_ARGS__)
#define DEFVAR_OPTDENSE_int32(name, title, desc, ...)   DEFVAR_int32(OPTDENSE, name, title, desc, __VA_ARGS__)
#define DEFVAR_OPTDENSE_uint32(name, title, desc, ...)  DEFVAR_uint32(OPTDENSE, name, title, desc, __VA_ARGS__)
#define DEFVAR_OPTDENSE_flags(name, title, desc, ...)   DEFVAR_flags(OPTDENSE, name, title, desc, __VA_ARGS__)
#define DEFVAR_OPTDENSE_float(name, title, desc, ...)   DEFVAR_float(OPTDENSE, name, title, desc, __VA_ARGS__)
#define DEFVAR_OPTDENSE_double(name, title, desc, ...)  DEFVAR_double(OPTDENSE, name, title, desc, __VA_ARGS__)

#define MDEFVAR_OPTDENSE_string(name, title, desc, ...) DEFVAR_string(OPTDENSE, name, title, desc, __VA_ARGS__)
#define MDEFVAR_OPTDENSE_bool(name, title, desc, ...)   DEFVAR_bool(OPTDENSE, name, title, desc, __VA_ARGS__)
#define MDEFVAR_OPTDENSE_int32(name, title, desc, ...)  DEFVAR_int32(OPTDENSE, name, title, desc, __VA_ARGS__)
#define MDEFVAR_OPTDENSE_uint32(name, title, desc, ...) DEFVAR_uint32(OPTDENSE, name, title, desc, __VA_ARGS__)
#define MDEFVAR_OPTDENSE_flags(name, title, desc, ...)  DEFVAR_flags(OPTDENSE, name, title, desc, __VA_ARGS__)
#define MDEFVAR_OPTDENSE_float(name, title, desc, ...)  DEFVAR_float(OPTDENSE, name, title, desc, __VA_ARGS__)
#define MDEFVAR_OPTDENSE_double(name, title, desc, ...) DEFVAR_double(OPTDENSE, name, title, desc, __VA_ARGS__)

namespace MVS {
DEFOPT_SPACE(OPTDENSE, _T("Dense"))

DEFVAR_OPTDENSE_uint32(nResolutionLevel, "Resolution Level", "How many times to scale down the images before dense reconstruction", "1")
MDEFVAR_OPTDENSE_uint32(nMaxResolution, "Max Resolution", "Do not scale images lower than this resolution", "3200")
MDEFVAR_OPTDENSE_uint32(nMinResolution, "Min Resolution", "Do not scale images lower than this resolution", "640")
DEFVAR_OPTDENSE_uint32(nMinViews, "Min Views", "minimum number of agreeing views to validate a depth", "2")
MDEFVAR_OPTDENSE_uint32(nMaxViews, "Max Views", "maximum number of neighbor images used to compute the depth-map for the reference image", "12")
DEFVAR_OPTDENSE_uint32(nMinViewsFuse, "Min Views Fuse", "minimum number of images that agrees with an estimate during fusion in order to consider it inlier", "2")
DEFVAR_OPTDENSE_uint32(nMinViewsFilter, "Min Views Filter", "minimum number of images that agrees with an estimate in order to consider it inlier", "2")
MDEFVAR_OPTDENSE_uint32(nMinViewsFilterAdjust, "Min Views Filter Adjust", "minimum number of images that agrees with an estimate in order to consider it inlier (0 - disabled)", "1")
MDEFVAR_OPTDENSE_uint32(nMinViewsTrustPoint, "Min Views Trust Point", "min-number of views so that the point is considered for approximating the depth-maps (<2 - random initialization)", "2")
MDEFVAR_OPTDENSE_uint32(nNumViews, "Num Views", "Number of views used for depth-map estimation (0 - all views available)", "0", "1", "4")
MDEFVAR_OPTDENSE_bool(bFilterAdjust, "Filter Adjust", "adjust depth estimates during filtering", "1")
MDEFVAR_OPTDENSE_bool(bAddCorners, "Add Corners", "add support points at image corners with nearest neighbor disparities", "1")
MDEFVAR_OPTDENSE_float(fViewMinScore, "View Min Score", "Min score to consider a neighbor images (0 - disabled)", "0")
MDEFVAR_OPTDENSE_float(fViewMinScoreRatio, "View Min Score Ratio", "Min score ratio to consider a neighbor images", "0.3")
MDEFVAR_OPTDENSE_float(fMinArea, "Min Area", "Min shared area for accepting the depth triangulation", "0.01")
MDEFVAR_OPTDENSE_float(fMinAngle, "Min Angle", "Min angle for accepting the depth triangulation", "3.0")
MDEFVAR_OPTDENSE_float(fOptimAngle, "Optim Angle", "Optimal angle for computing the depth triangulation", "10.0")
MDEFVAR_OPTDENSE_float(fMaxAngle, "Max Angle", "Max angle for accepting the depth triangulation", "65.0")
MDEFVAR_OPTDENSE_float(fDescriptorMinMagnitudeThreshold, "Descriptor Min Magnitude Threshold", "minimum texture variance accepted when matching two patches (0 - disabled)", "0.01")
MDEFVAR_OPTDENSE_float(fDepthDiffThreshold, "Depth Diff Threshold", "maximum variance allowed for the depths during refinement", "0.01")
MDEFVAR_OPTDENSE_float(fNormalDiffThreshold, "Normal Diff Threshold", "maximum variance allowed for the normal during fusion (degrees)", "25")
MDEFVAR_OPTDENSE_float(fPairwiseMul, "Pairwise Mul", "pairwise cost scale to match the unary cost", "0.3")
MDEFVAR_OPTDENSE_float(fOptimizerEps, "Optimizer Eps", "MRF optimizer stop epsilon", "0.001")
MDEFVAR_OPTDENSE_int32(nOptimizerMaxIters, "Optimizer Max Iters", "MRF optimizer max number of iterations", "80")
MDEFVAR_OPTDENSE_uint32(nSpeckleSize, "Speckle Size", "maximal size of a speckle (small speckles get removed)", "100")
MDEFVAR_OPTDENSE_uint32(nIpolGapSize, "Interpolate Gap Size", "interpolate small gaps (left<->right, top<->bottom)", "7")
MDEFVAR_OPTDENSE_string(nIgnoreMaskLabel, "Ignore mask labels", "Label values to ignore in the segmentation masks (separated by comma)", "-1")
MDEFVAR_OPTDENSE_bool(nUseSemantic, "Use semantic label masks", "Use semantic label masks", "1")
MDEFVAR_OPTDENSE_uint32(nOptimize, "Optimize", "should we filter the extracted depth-maps?", "0")  // see DepthFlags
MDEFVAR_OPTDENSE_uint32(photo2geo, "photo2geo", "should we filter the extracted depth-maps?", "2") // see DepthFlags
MDEFVAR_OPTDENSE_uint32(usepartconsistency, "usepartconsistency", "should we filter the extracted depth-maps?", "2") // see DepthFlags
MDEFVAR_OPTDENSE_uint32(usegeoconsistency, "usegeoconsistency", "should we filter the extracted depth-maps?", "2")  // see DepthFlags
MDEFVAR_OPTDENSE_uint32(initTriangulate, "initTriangulate", "should we filter the extracted depth-maps?", "2")  // see DepthFlags
MDEFVAR_OPTDENSE_uint32(viewspread, "viewspread", "should we filter the extracted depth-maps?", "2")  // see DepthFlags
MDEFVAR_OPTDENSE_uint32(opticalflow, "opticalflow", "should we filter the extracted depth-maps?", "2")  // see DepthFlags
MDEFVAR_OPTDENSE_float(txthreshold, "txthreshold", "Number of iterations for depth-map refinement", "2")
MDEFVAR_OPTDENSE_float(txthreshold2, "txthreshold2", "Number of iterations for depth-map refinement", "2")
MDEFVAR_OPTDENSE_float(para_part, "para_part", "Number of iterations for depth-map refinement", "0.3")
MDEFVAR_OPTDENSE_float(para_part2, "para_part2", "Number of iterations for depth-map refinement", "0.1")
MDEFVAR_OPTDENSE_float(para_tapa, "para_part", "Number of iterations for depth-map refinement", "0.3")
MDEFVAR_OPTDENSE_float(para_tapa2, "para_part", "Number of iterations for depth-map refinement", "0.3")
MDEFVAR_OPTDENSE_float(para_prior, "para_prior2", "Number of iterations for depth-map refinement", "0.3")
MDEFVAR_OPTDENSE_float(para_prior2, "para_prior2", "Number of iterations for depth-map refinement", "0.3")
MDEFVAR_OPTDENSE_float(photometric_flow, "photometric_flow", "Number of iterations for depth-map refinement", "0.5")
MDEFVAR_OPTDENSE_float(maxgeo_proportion, "maxgeo_proportion", "Number of iterations for depth-map refinement", "0.5")
MDEFVAR_OPTDENSE_uint32(nEstimateColors, "Estimate Colors", "should we estimate the colors for the dense point-cloud?", "2", "0", "1")
MDEFVAR_OPTDENSE_uint32(nEstimateNormals, "Estimate Normals", "should we estimate the normals for the dense point-cloud?", "0", "1", "2")
MDEFVAR_OPTDENSE_uint32(ProjectLabels, "Project Labels", "should we estimate the labels for the dense point-cloud?", "0", "1")
MDEFVAR_OPTDENSE_float(fNCCThresholdKeep, "NCC Threshold Keep", "Maximum 1-NCC score accepted for a match", "0.55", "0.3")
MDEFVAR_OPTDENSE_uint32(nEstimationIters, "Estimation Iters", "Number of iterations for depth-map refinement", "1")
MDEFVAR_OPTDENSE_uint32(nEstimationIters_external, "Estimation Iters", "Number of iterations for depth-map refinement", "4")
MDEFVAR_OPTDENSE_uint32(nRandomIters, "Random Iters", "Number of iterations for random assignment per pixel", "6")
MDEFVAR_OPTDENSE_uint32(nRandomMaxScale, "Random Max Scale", "Maximum number of iterations to skip during random assignment", "2")
MDEFVAR_OPTDENSE_float(fRandomDepthRatio, "Random Depth Ratio", "Depth range ratio of the current estimate for random plane assignment", "0.003", "0.004")
MDEFVAR_OPTDENSE_float(fRandomAngle1Range, "Random Angle1 Range", "Angle 1 range for random plane assignment (degrees)", "16.0", "20.0")
MDEFVAR_OPTDENSE_float(fRandomAngle2Range, "Random Angle2 Range", "Angle 2 range for random plane assignment (degrees)", "10.0", "12.0")
MDEFVAR_OPTDENSE_float(fRandomSmoothDepth, "Random Smooth Depth", "Depth variance used during neighbor smoothness assignment (ratio)", "0.02")
MDEFVAR_OPTDENSE_float(fRandomSmoothNormal, "Random Smooth Normal", "Normal variance used during neighbor smoothness assignment (degrees)", "13")
MDEFVAR_OPTDENSE_float(fRandomSmoothBonus, "Random Smooth Bonus", "Score factor used to encourage smoothness (1 - disabled)", "0.93")
MDEFVAR_OPTDENSE_float(fSemanticConsistencyMul, "Semantic Consistency Multiplier", "Weight of semantic prior", "0.1")
MDEFVAR_OPTDENSE_float(fsigmaTexture, "sigma texture", "Textureness coefficient sigma", "0.05")
MDEFVAR_OPTDENSE_float(fsigmaPrior, "sigma prior", "Prior coefficient sigma", "0.2")
MDEFVAR_OPTDENSE_float(fransacEpsilonMul, "RANSAC epsilon multiplier", "avg_spacing*fransacEpsilonMul = maximum Euclidean distance between a point and a shape", "2")
MDEFVAR_OPTDENSE_float(fransacClusterMul, "RANSAC cluster multiplier", "avg_spacing*fransacClusterMul = maximum Euclidean distance between points to be clustered", "10")
MDEFVAR_OPTDENSE_float(fransacMinPointsDiv, "RANSAC min points divider", "points.size()/fransacMinPointsDiv = Detect shapes with at least size / n points.", "80")

MDEFVAR_OPTDENSE_float(ransacprobability, "RANSAC probability", "probability.", "0.01")

MDEFVAR_OPTDENSE_float(depthweight, "normalweight probability", "depthweight.", "1.0")
MDEFVAR_OPTDENSE_float(normalweight, "normalweight probability", "normalweight.", "1.0")


MDEFVAR_OPTDENSE_uint32(adapthalfwin, "adapthalfwin", "should we filter the extracted depth-maps?", "5") 		 // see DepthFlags
MDEFVAR_OPTDENSE_uint32(propagatehalfwin, "propagatehalfwin", "should we filter the extracted depth-maps?", "1") // see DepthFlags
MDEFVAR_OPTDENSE_uint32(propagatestep, "propagatestep", "should we filter the extracted depth-maps?", "4") // see DepthFlags



}



// S T R U C T S ///////////////////////////////////////////////////

// return normal in world-space for the given pixel
// the 3D points can be precomputed and passed here
void DepthData::GetNormal(const ImageRef& ir, Point3f& N, const TImage<Point3f>* pPointMap) const
{
	ASSERT(!IsEmpty());
	ASSERT(depthMap(ir) > 0);
	const Camera& camera = images.First().camera;
	if (!normalMap.empty()) {
		// set available normal
		N = camera.R.t()*Cast<REAL>(normalMap(ir));
		return;
	}
	// estimate normal based on the neighbor depths
	const int nPointsStep = 2;
	const int nPointsHalf = 2;
	const int nPoints = 2*nPointsHalf+1;
	const int nWindowHalf = nPointsHalf*nPointsStep;
	const int nWindow = 2*nWindowHalf+1;
	const Image8U::Size size(depthMap.size());
	const ImageRef ptCorner(ir.x-nWindowHalf, ir.y-nWindowHalf);
	const ImageRef ptCornerRel(ptCorner.x>=0?0:-ptCorner.x, ptCorner.y>=0?0:-ptCorner.y);
	Point3Arr points(1, nPoints*nPoints);
	if (pPointMap) {
		points[0] = (*pPointMap)(ir);
		for (int j=ptCornerRel.y; j<nWindow; j+=nPointsStep) {
			const int y = ptCorner.y+j;
			if (y >= size.height)
				break;
			for (int i=ptCornerRel.x; i<nWindow; i+=nPointsStep) {
				const int x = ptCorner.x+i;
				if (x >= size.width)
					break;
				if (x==ir.x && y==ir.y)
					continue;
				if (depthMap(y,x) > 0)
					points.Insert((*pPointMap)(y,x));
			}
		}
	} else {
		points[0] = camera.TransformPointI2C(Point3(ir.x,ir.y,depthMap(ir)));
		for (int j=ptCornerRel.y; j<nWindow; j+=nPointsStep) {
			const int y = ptCorner.y+j;
			if (y >= size.height)
				break;
			for (int i=ptCornerRel.x; i<nWindow; i+=nPointsStep) {
				const int x = ptCorner.x+i;
				if (x >= size.width)
					break;
				if (x==ir.x && y==ir.y)
					continue;
				const Depth d = depthMap(y,x);
				if (d > 0)
					points.Insert(camera.TransformPointI2C(Point3(x,y,d)));
			}
		}
	}
	if (points.GetSize() < 3) {
		N = normalized(-points[0]);
		return;
	}
	Plane plane;
	if (EstimatePlaneThLockFirstPoint(points, plane, 0, NULL, 20) < 3) {
		N = normalized(-points[0]);
		return;
	}
	ASSERT(ISEQUAL(plane.m_vN.norm(),REAL(1)));
	// normal is defined up to sign; pick the direction that points to the camera
	if (plane.m_vN.dot((const Point3::EVec)points[0]) > 0)
		plane.Negate();
	N = camera.R.t()*Point3(plane.m_vN);
}
void DepthData::GetNormal(const Point2f& pt, Point3f& N, const TImage<Point3f>* pPointMap) const
{
	const ImageRef ir(ROUND2INT(pt));
	GetNormal(ir, N, pPointMap);
} // GetNormal
/*----------------------------------------------------------------*/


// apply mask to the depth map
void DepthData::ApplyIgnoreMask(const BitMatrix& mask)
{
	ASSERT(IsValid() && !IsEmpty() && mask.size() == depthMap.size());
	for (int r=0; r<depthMap.rows; ++r) {
		for (int c=0; c<depthMap.cols; ++c) {
			if (mask.isSet(r,c))
				continue;
			// discard depth-map section ignored by mask
			depthMap(r,c) = 0;
			if (!normalMap.empty())
				normalMap(r,c) = Normal::ZERO;
			if (!confMap.empty())
				confMap(r,c) = 0;
		}
	}
} // ApplyIgnoreMask
/*----------------------------------------------------------------*/



bool DepthData::Save(const String& fileName) const
{
	ASSERT(IsValid() && !depthMap.empty() && !confMap.empty());
	const String fileNameTmp(fileName+".tmp"); {
		// serialize out the current state
		IIndexArr IDs(0, images.size());
		for (const ViewData& image: images)
			IDs.push_back(image.GetID());
		const ViewData& image0 = GetView();
		if (!ExportDepthDataRaw(fileNameTmp, image0.pImageData->name, IDs, depthMap.size(), image0.camera.K, image0.camera.R, image0.camera.C, dMin, dMax, depthMap, normalMap, confMap))
			return false;
	}
	if (!File::renameFile(fileNameTmp, fileName)) {
		DEBUG_EXTRA("error: can not access dmap file '%s'", fileName.c_str());
		File::deleteFile(fileNameTmp);
		return false;
	}
	return true;
}
bool DepthData::Load(const String& fileName)
{
	ASSERT(IsValid());
	// serialize in the saved state
	String imageFileName;
	IIndexArr IDs;
	cv::Size imageSize;
	Camera camera;
	if (!ImportDepthDataRaw(fileName, imageFileName, IDs, imageSize, camera.K, camera.R, camera.C, dMin, dMax, depthMap, normalMap, confMap))
		return false;
	ASSERT(IDs.size() == images.size());
	ASSERT(IDs.front() == GetView().GetID());
	ASSERT(depthMap.size() == imageSize);
	return true;
}
/*----------------------------------------------------------------*/


unsigned DepthData::GetRef()
{
	Lock l(cs);
	return references;
}
unsigned DepthData::IncRef(const String& fileName)
{
	Lock l(cs);
	ASSERT(!IsEmpty() || references==0);
	if (IsEmpty() && !Load(fileName))
		return 0;
	return ++references;
}
unsigned DepthData::DecRef()
{
	Lock l(cs);
	ASSERT(references>0);
	if (--references == 0)
		Release();
	return references;
}
/*----------------------------------------------------------------*/



// S T R U C T S ///////////////////////////////////////////////////

// try to load and apply mask to the depth map;
// the mask marks as false pixels that should be ignored
bool DepthEstimator::ImportIgnoreMask(const Image& image0, const Image8U::Size& size, BitMatrix& bmask, String& nIgnoreMaskLabel)
{
	CLISTDEF2(String) ignoreMaskLabels;
	Util::strSplit(nIgnoreMaskLabel, ",", ignoreMaskLabels);

	ASSERT(image0.IsValid() && !image0.image.empty());
	if (image0.maskName.empty())
		return false;
	if (ignoreMaskLabels.empty())
		return false;
		
	Image16U mask;
	if (!mask.Load(image0.maskName)) {
		DEBUG("warning: can not load the segmentation mask '%s'", image0.maskName.c_str());
		return false;
	}

	cv::resize(mask, mask, size, 0, 0, cv::INTER_NEAREST);
	bmask.create(size);
	bmask.memset(0xFF);
	for (int r=0; r<size.height; ++r) {
		for (int c=0; c<size.width; ++c) {
			for (int i=0; i < ignoreMaskLabels.size(); i++){
				if (mask(r,c) == atoi(ignoreMaskLabels[i].c_str()))
					bmask.unset(r,c);
			}
		}
	}
	return true;
} // ImportIgnoreMask

// create the map for converting index to matrix position
//                         1 2 3
//  1 2 4 7 5 3 6 8 9 -->  4 5 6
//                         7 8 9
void DepthEstimator::MapMatrix2ZigzagIdx(const Image8U::Size& size, DepthEstimator::MapRefArr& coords, const BitMatrix& mask, int rawStride)
{
	typedef DepthEstimator::MapRef MapRef;
	const int w = size.width;
	const int w1 = size.width-1;
	coords.Empty();
	coords.Reserve(size.area());
	for (int dy=0, h=rawStride; dy<size.height; dy+=h) {
		if (h*2 > size.height - dy)
			h = size.height - dy;
		int lastX = 0;
		MapRef x(MapRef::ZERO);
		for (int i=0, ei=w*h; i<ei; ++i) {
			const MapRef pt(x.x, x.y+dy);
			if (mask.empty() || mask.isSet(pt))
				coords.Insert(pt);
			if (x.x-- == 0 || ++x.y == h) {
				if (++lastX < w) {
					x.x = lastX;
					x.y = 0;
				} else {
					x.x = w1;
					x.y = lastX - w1;
				}
			}
		}
	}
}

// replace POWI(0.5f, (int)idxScaleRange):      0    1      2       3       4         5         6           7           8             9             10              11
const float DepthEstimator::scaleRanges[12] = {1.f, 0.5f, 0.25f, 0.125f, 0.0625f, 0.03125f, 0.015625f, 0.0078125f, 0.00390625f, 0.001953125f, 0.0009765625f, 0.00048828125f};

DepthEstimator::DepthEstimator(
	unsigned nIter,int nIter_external, DepthData& _depthData0, DepthDataArr& _arrDepthData0, volatile Thread::safe_t& _idx,
	#if DENSE_NCC == DENSE_NCC_WEIGHTED
	WeightMap& _weightMap0,
	#else
	const Image64F& _image0Sum,
	#endif
	const MapRefArr& _coords)
	:
	#ifndef _RELEASE
	rnd(SEACAVE::Random::default_seed),
	#endif
	idxPixel(_idx),
	neighbors(0,2),
	#if DENSE_SMOOTHNESS != DENSE_SMOOTHNESS_NA
	neighborsClose(0,4),
	#endif
	arrDepthData0(_arrDepthData0) ,
	scores(_depthData0.images.size()-1),
	depthMap0(_depthData0.depthMap), normalMap0(_depthData0.normalMap), confMap0(_depthData0.confMap),
	#if DENSE_NCC == DENSE_NCC_WEIGHTED
	weightMap0(_weightMap0),
	#endif
	nIteration(nIter),
	nIteration_external(nIter_external),	
	images(InitImages(_depthData0)), image0(_depthData0.images[0]),image1(_depthData0.images[1]),
	#if DENSE_NCC != DENSE_NCC_WEIGHTED
	image0Sum(_image0Sum),
	#endif
	coords(_coords), size(_depthData0.images.First().image.size()),
	dMin(_depthData0.dMin), dMax(_depthData0.dMax),delta_c2pmax(_depthData0.delta_c2pmax),
	dMinSqr(SQRT(_depthData0.dMin)), dMaxSqr(SQRT(_depthData0.dMax)),
	dir(nIter%2 ? RB2LT : LT2RB),
	#if DENSE_AGGNCC == DENSE_AGGNCC_NTH
	idxScore((_depthData0.images.size()-1)/3),
	#elif DENSE_AGGNCC == DENSE_AGGNCC_MINMEAN
	idxScore(_depthData0.images.size()<=2 ? 0u : 1u),
	#endif
	smoothBonusDepth(1.f-OPTDENSE::fRandomSmoothBonus), smoothBonusNormal((1.f-OPTDENSE::fRandomSmoothBonus)*0.96f),
	smoothSigmaDepth(-1.f/(2.f*SQUARE(OPTDENSE::fRandomSmoothDepth))), // used in exp(-x^2 / (2*(0.02^2)))
	smoothSigmaNormal(-1.f/(2.f*SQUARE(FD2R(OPTDENSE::fRandomSmoothNormal)))), // used in exp(-x^2 / (2*(0.22^2)))
	thMagnitudeSq(OPTDENSE::fDescriptorMinMagnitudeThreshold>0?SQUARE(OPTDENSE::fDescriptorMinMagnitudeThreshold):-1.f),
	angle1Range(FD2R(OPTDENSE::fRandomAngle1Range)),
	angle2Range(FD2R(OPTDENSE::fRandomAngle2Range)),
	thConfSmall(OPTDENSE::fNCCThresholdKeep*0.2f),
	thConfBig(OPTDENSE::fNCCThresholdKeep*0.4f),
	thConfRand(OPTDENSE::fNCCThresholdKeep*0.9f),
	thRobust(OPTDENSE::fNCCThresholdKeep*1.2f) // default is 1.2f
	#if DENSE_REFINE == DENSE_REFINE_EXACT
	, thPerturbation(1.f/POW(2.f,float(nIter+1)))
	#endif
{
	ASSERT(_depthData0.images.size() >= 1);
}

// center a patch of given size on the segment
bool DepthEstimator::PreparePixelPatch(const ImageRef& x)
{
	x0 = x;
	return image0.image.isInside(ImageRef(x.x-nSizeHalfWindow, x.y-nSizeHalfWindow)) &&
	       image0.image.isInside(ImageRef(x.x+nSizeHalfWindow, x.y+nSizeHalfWindow));
}

// fetch the patch pixel values in the main image
bool DepthEstimator::FillPixelPatch()
{

	uint id_0 = image0.GetID(); 	  	
	float tx(arrDepthData0[id_0].graMap(x0)); 
	adapthalfwin = OPTDENSE::adapthalfwin ;  
	if(tx > 100){  
		adapthalfwin = 5 ;	
	}else{
		adapthalfwin = OPTDENSE::adapthalfwin ;
	}
	const int pointnum = SQUARE((adapthalfwin*2+nSizeStep)/nSizeStep)*TexelChannels ;	
	/*******************************************************/


	#if DENSE_NCC != DENSE_NCC_WEIGHTED
	// const float mean(GetImage0Sum(x0)/nTexels);
	const float mean(GetImage0Sum(x0)/pointnum);	


	normSq0 = 0;
	float* pTexel0 = texels0.data();
	// for (int i=-nSizeHalfWindow; i<=nSizeHalfWindow; i+=nSizeStep)
	// 	for (int j=-nSizeHalfWindow; j<=nSizeHalfWindow; j+=nSizeStep)
	for (int i=-adapthalfwin; i<=adapthalfwin; i+=nSizeStep)	
		for (int j=-adapthalfwin; j<=adapthalfwin; j+=nSizeStep)	
			normSq0 += SQUARE(*pTexel0++ = image0.image(x0.y+i, x0.x+j)-mean);
	#else
	Weight& w = weightMap0[x0.y*image0.image.width()+x0.x];
	if (w.normSq0 == 0) {
		w.sumWeights = 0;
		int n = 0;
		const float colCenter = image0.image(x0);
		// for (int i=-nSizeHalfWindow; i<=nSizeHalfWindow; i+=nSizeStep) {
		// 	for (int j=-nSizeHalfWindow; j<=nSizeHalfWindow; j+=nSizeStep) {
		for (int i=-adapthalfwin; i<=adapthalfwin; i+=nSizeStep) {	
			for (int j=-adapthalfwin; j<=adapthalfwin; j+=nSizeStep) {	
				Weight::Pixel& pw = w.weights[n++];
				w.normSq0 +=
					(pw.tempWeight = image0.image(x0.y+i, x0.x+j)) *
					(pw.weight = GetWeight(ImageRef(j,i), colCenter));
				w.sumWeights += pw.weight;
			}
		}
		// ASSERT(n == nTexels);
		ASSERT(n == pointnum);	

		const float tm(w.normSq0/w.sumWeights);
		w.normSq0 = 0;
		n = 0;
		do {
			Weight::Pixel& pw = w.weights[n];
			const float t(pw.tempWeight - tm);
			w.normSq0 += (pw.tempWeight = pw.weight * t) * t;
		} while (++n < pointnum);	
		// } while (++n < nTexels);
	}
	normSq0 = w.normSq0;
	#endif

	if (image0.depthMapPrior.empty() || image0.depthMapPrior(x0) == 0) // If we can trust the priors return all the pixels
	{
		if (normSq0 < thMagnitudeSq)
		;
			// return false;
	}
	reinterpret_cast<Point3&>(X0) = image0.camera.TransformPointI2C(Cast<REAL>(x0));
	return true;
}

// compute pixel's NCC score in the given target image
float DepthEstimator::ScorePixelImage(int idxView, const ViewData& image1, Depth depth, const Normal& normal)
{	
	// center a patch of given size on the segment and fetch the pixel values in the target image
	Matrix3x3f H(ComputeHomographyMatrix(image1, depth, normal));
	Point3f X;
	// ProjectVertex_3x3_2_3(H.val, Point2f(float(x0.x-nSizeHalfWindow),float(x0.y-nSizeHalfWindow)).ptr(), X.ptr());
	ProjectVertex_3x3_2_3(H.val, Point2f(float(x0.x-adapthalfwin),float(x0.y-adapthalfwin)).ptr(), X.ptr());	

	Point3f baseX(X);

	Point3f X1 ;
	ProjectVertex_3x3_2_3(H.val, Point2f(float(x0.x),float(x0.y)).ptr(), X1.ptr());
	const Point2f x1_f(X1) ; 
	Point2i x1_i; 			
	x1_i.x = (int)(x1_f.x) ;
	x1_i.y = (int)(x1_f.y) ;
	// std::cout<<X1.x<<" "<<X1.y<<" "<<"Z="<<X1.z<<" "<<std::endl;
	H *= float(nSizeStep);
	int n(0);
	float sum(0);
	#if DENSE_NCC != DENSE_NCC_DEFAULT
	float sumSq(0), num(0);
	#endif
	#if DENSE_NCC == DENSE_NCC_WEIGHTED
	const Weight& w = weightMap0[x0.y*image0.image.width()+x0.x];
	#endif
	// for (int i=-nSizeHalfWindow; i<=nSizeHalfWindow; i+=nSizeStep) {
	// 	for (int j=-nSizeHalfWindow; j<=nSizeHalfWindow; j+=nSizeStep) {
	for (int i=-adapthalfwin; i<=adapthalfwin; i+=nSizeStep) {	
		for (int j=-adapthalfwin; j<=adapthalfwin; j+=nSizeStep) {	
			const Point2f pt(X);
			if (!image1.view.image.isInsideWithBorder<float,1>(pt))
				return thRobust;
			const float v(image1.view.image.sample(pt));
			#if DENSE_NCC == DENSE_NCC_FAST
			sum += v;
			sumSq += SQUARE(v);
			num += texels0(n++)*v;
			#elif DENSE_NCC == DENSE_NCC_WEIGHTED
			const Weight::Pixel& pw = w.weights[n++];
			const float vw(v*pw.weight);
			sum += vw;
			sumSq += v*vw;
			num += v*pw.tempWeight;
			#else
			sum += texels1(n++)=v;
			#endif
			X.x += H[0]; X.y += H[3]; X.z += H[6];
		}
		baseX.x += H[1]; baseX.y += H[4]; baseX.z += H[7];
		X = baseX;
	}
	// ASSERT(n == nTexels);
	ASSERT(n == pointnum);	

	// score similarity of the reference and target texture patches
	#if DENSE_NCC == DENSE_NCC_FAST
	const float normSq1(sumSq-SQUARE(sum/nSizeWindow));
	#elif DENSE_NCC == DENSE_NCC_WEIGHTED
	const float normSq1(sumSq-SQUARE(sum)/w.sumWeights);
	#else
	const float normSq1(normSqDelta<float,float,nTexels>(texels1.data(), sum/(float)nTexels));
	#endif
	const float nrmSq(normSq0*normSq1);
	if (nrmSq <= 0.f)
		return thRobust;
	#if DENSE_NCC == DENSE_NCC_DEFAULT
	const float num(texels0.dot(texels1));
	#endif
	const float ncc(CLAMP(num/SQRT(nrmSq), -1.f, 1.f));
	float score(0);
	float score_ncc(1.f-ncc);
	float score_geo(1.5);
	float score_priors(0);
	float score_gra(1.5);
	float score_flow(0);


	#if DENSE_SMOOTHNESS != DENSE_SMOOTHNESS_NA
	// encourage smoothness
	for (const NeighborEstimate& neighbor: neighborsClose) {
		#if DENSE_SMOOTHNESS == DENSE_SMOOTHNESS_PLANE
		const float factorDepth(DENSE_EXP(SQUARE(plane.Distance(neighbor.X)/depth) * smoothSigmaDepth));
		#else
		const float factorDepth(DENSE_EXP(SQUARE((depth-neighbor.depth)/depth) * smoothSigmaDepth));
		#endif
		const float factorNormal(DENSE_EXP(SQUARE(ACOS(ComputeAngle<float,float>(normal.ptr(), neighbor.normal.ptr()))) * smoothSigmaNormal));
		score_ncc *= (1.f - smoothBonusDepth * factorDepth) * (1.f - smoothBonusNormal * factorNormal);
	}
	#endif
	


	const int x1_width  = arrDepthData0[0].depthMap.width() ;
	const int x1_height = arrDepthData0[0].depthMap.height() ;
	float x1_maxdistand = SQRT(SQUARE(x1_width/2)+SQUARE(x1_height/2));
	if(nIteration_external>=OPTDENSE::photo2geo)
	{
		
		
		if ( (x1_i.x>=x1_width  ) || (x1_i.y>=x1_height) || (x1_i.x<0) || (x1_i.y<0) ) 
		{
			score_geo= 1 ;
			// std::cout<<"1"<<std::endl;
		}
		else
		{
			// std::cout<<"2"<<std::endl;

			uint id_0 = image0.GetID(); 	  	
			uint id_1 = image1.view.GetID() ;   


			DepthData& depthData_image0 = arrDepthData0[id_0] ;
			DepthData& depthData_image1 = arrDepthData0[id_1] ;
			
			// std::cout<<"4"<<std::endl;

			const Normal normal_img0 = depthData_image0.normalMap(x0);

			const Depth depth_img1 = depthData_image1.depthMap(x1_i);
			const Normal normal_img1 = depthData_image1.normalMap(x1_i);

			float image_geo_standard(0);

			if(depth_img1 == 0)
			{
				score_geo = 1 ;
				// std::cout<<"3"<<std::endl;

			}
			else
			{
				Matrix3x3f H1(ComputeHomoMatrix(image1, depth_img1 , normal_img1, x1_i) ); 
				Point3f X1_0;
				ProjectVertex_3x3_2_3(H1.val, Point2f(float(x1_i.x),float(x1_i.y)).ptr(), X1_0.ptr());
				Point2f x0_(X1_0); 
				Point2i x0_int ;   
				x0_int.x = (int)(x0_.x) ; 
				x0_int.y = (int)(x0_.y) ;
				Matrix3x3f F(ComputeFundamentalMatrix(image1));
				Vec3f X1V(x1_i.x,x1_i.y,1) ;
				Vec3f F_tmp = F*X1V;
				// std::cout<< F[0]<<"  "<< F[1]<<"  "<< F[2]<<"  "<< F[3]<< "  "<<F[4]<<"  "<< F[5]<<"  "<< F[6]<<"  "<< F[7]<<"  "<< F[8]<< std::endl;
				// std::cout<<X1V[0]<<"  "<<X1V[1]<<"  "<<X1V[2]<<std::endl;
				// std::cout<<F_tmp[0]<<"  "<<F_tmp[1]<<"  "<<F_tmp[2]<<std::endl;
				double A = F_tmp[0]+F_tmp[1]+F_tmp[2];
				double B = F_tmp[3]+F_tmp[4]+F_tmp[5];
				double C = F_tmp[6]+F_tmp[7]+F_tmp[8];
				double dis_tmp = abs(A*x0.x+B*x0.y+C);
				double dis = dis_tmp/SQRT(A*A+B*B);
				
				if (!image1.view.image.isInsideWithBorder<float,1>(x0_int)) 
				{
					score_geo = 1 ;
					// std::cout<<"4"<<std::endl;

				}
				else
				{
			
					float  geox = x0.x - x0_.x ;
					float  geoy = x0.y - x0_.y ;
					float  disGeo = (geox*geox) + ( geoy*geoy ) ;
					float  image_geo =	SQRT( disGeo );
					double dotSum=normal_img0.dot(normal_img1);
					double normFirst=cv::norm(normal_img0);
					double normSecond=cv::norm(normal_img1); 
					if(image_geo > x1_maxdistand){
						score_geo = 1 ;
						// std::cout<<"5"<<std::endl;

					}
					else{
						if(dis == 0){
							float ratio = ((C/A)*geox - (C/B)*geoy)/(SQRT(SQUARE(C/B)+SQUARE(C/A))+image_geo);
							image_geo_standard = image_geo*ratio;
						}
						else{
							image_geo_standard = dis ;
						}
						if (image_geo < image_geo_standard*OPTDENSE::maxgeo_proportion)
							// score_geo =2* (image_geo*1.0/image_geo_standard);
							score_geo = (image_geo*1.0/image_geo_standard) + (1- abs(dotSum/(normFirst*normSecond)));
						else{
							score_geo = 2 ;

						}  
							

					}
					
				}
			}
		}
	}


	



	// float photometric_flow(0.5);
	// float scoredis(0);
	Point2f vector_x0 ;
	Point2f vector_x1 ;
	vector_x0.x = x1_f.x - x0.x; 
	vector_x0.y = x1_f.y - x0.y; 
	uint id_0 = image0.GetID();
	DepthData& depthData_image0 = arrDepthData0[id_0] ;
	if(OPTDENSE::opticalflow){
		if(idxView == 1){
			if ( (x1_i.x>=x1_width  ) || (x1_i.y>=x1_height) || (x1_i.x<0) || (x1_i.y<0) ) 
			{
				score_flow = 1 ;
			}
			else
			{
				vector_x1.x = arrDepthData0[id_0].flow_images[idxView].x0.x ;
				vector_x1.y = arrDepthData0[id_0].flow_images[idxView].x0.y ;
				// cv::imwrite("1.png", depthData_image0.images[idxView].pImageData->image) ;
				// cvWaitKey(2000);
				double dotSum=vector_x0.dot(vector_x1);
				double normFirst=cv::norm(vector_x0);
				double normSecond=cv::norm(vector_x1); 
				double length_ratio=0;
				if(normFirst==0 && normSecond!=0){
					score_flow = 1;
				}
				if(normFirst!=0 && normSecond==0){
					score_flow = 0;
				}
				if(normFirst!=0 && normSecond!=0){
					if (normFirst>x1_maxdistand||normFirst==x1_maxdistand){
						score_flow = 2;
					}
					else{
						if(normFirst > normSecond){
							length_ratio = normSecond/normFirst;
						}
						else{
							length_ratio = normFirst/normSecond;

						}

						float score_flow = (1- abs(dotSum/(normFirst*normSecond)) + length_ratio) ;
						//float score_flowtmp =2* (1- abs(dotSum/(normFirst*normSecond)));
					}
				}
			}	

		}
	}
	// float photometric_weight(1- OPTDENSE::photometric_flow);




	// DepthData& depthData_image0 = arrDepthData0[id_0] ;
	int graHalfWindowSize = 2; 
	// const int x1_width  = arrDepthData0[0].depthMap.width() ;
	// const int x1_height = arrDepthData0[0].depthMap.height() ;
	std::vector<Point2i> pointlist ;
	if ( (x0.x<x1_width-graHalfWindowSize) && (x0.y<x1_height-graHalfWindowSize) && (x0.x>graHalfWindowSize-1) && (x0.y>graHalfWindowSize-1) ) 
	{
		for(int i=x0.x-graHalfWindowSize ; i<x0.x+graHalfWindowSize  ; i++)
		{
			for(int j=x0.y-graHalfWindowSize ; j<x0.y+graHalfWindowSize ; j++)
			{
				Point2i x1; x1.x=i ; x1.y=j ;
				pointlist.push_back(x1);
			}
		}
	}
	float diff(0) ;   
	float ndiff(0);   
	REAL c2pdiff(0); 
	int valid_num(0); 
	REAL c2p_cur(normal.dot(image0.camera.TransformPointI2C(Point3(x0, depth)))); 
	for(int i=0 ; i<pointlist.size() ;i++ )
	{
	
		if(depthData_image0.depthMap(pointlist[i]) == 0)
		{ continue ;}

		
		diff += abs(depth-depthData_image0.depthMap(pointlist[i])); 

		ndiff += abs(normal.x-depthData_image0.normalMap(pointlist[i]).x) ;
		ndiff += abs(normal.y-depthData_image0.normalMap(pointlist[i]).y) ;
		ndiff += abs(normal.z-depthData_image0.normalMap(pointlist[i]).z) ; 
        
		REAL c2p_nx(normal.dot(image0.camera.TransformPointI2C(Point3(pointlist[i], depthData_image0.depthMap(pointlist[i]))))); 
        c2pdiff += abs(c2p_cur - c2p_nx);

		valid_num++; 
	}
	
	if(valid_num != 0)
	{
		diff = diff*1.0 / valid_num ;
		ndiff = ndiff*1.0 / valid_num / 3 ;  ndiff = ndiff*2 ;
		c2pdiff = c2pdiff*1.0/valid_num;
	}
	
	depthData_image0.diffMap(x0)=diff ;
	depthData_image0.ndiffMap(x0)=ndiff ;



	// if(nIteration_external>=OPTDENSE::photo2geo)
	if(nIteration_external>=1)
	{
		
		float dmax = depthData_image0.dMax ;
		float dmin = depthData_image0.dMin ;
        
		
		if ( (x0.x<x1_width-graHalfWindowSize) && (x0.y<x1_height-graHalfWindowSize) && (x0.x>graHalfWindowSize-1) && (x0.y>graHalfWindowSize-1) ) 
		{
			
			float diff_max = dmax*0.5 ;
			float diff_c2p_max = delta_c2pmax*0.5;
			if(delta_c2pmax == 0)
			std::cout<<"error"<<std::endl;
			if(diff > diff_max)
			{ diff=2; }
			else
			{diff = 2*diff*1.0 / diff_max ;}

			if(c2pdiff > diff_c2p_max)
			{ c2pdiff=2; }
			else
			{c2pdiff = 2*c2pdiff*1.0 / diff_c2p_max ;}


			score_gra = (diff+ndiff+c2pdiff)/3;

			depthData_image0.scoreGraMap(x0) = score_gra ;
		}
	}

	if(nIteration_external<OPTDENSE::photo2geo)
	// if(nIteration_external < 1)
	score = score_ncc ;

	if(nIteration_external>=OPTDENSE::photo2geo)
	// if(nIteration_external>=1)
	{
        float para_tapa(0);  
		float para_part(0);  
		//double score_part(0);   
		uint id_0 = image0.GetID(); 	  	
        float tx(arrDepthData0[id_0].graMap(x0));       
		// float num_nx(0);                   
		// const float colCenter = image0.image(x0); 
		if(OPTDENSE::usepartconsistency)
		{
			if(tx < OPTDENSE::txthreshold){         
				para_part = OPTDENSE::para_part;   
			}
			else{
				if(tx < OPTDENSE::txthreshold2){        
					para_part = OPTDENSE::para_part2; 
				}
			}
		}
        if(OPTDENSE::usegeoconsistency){     
		
			if(tx < OPTDENSE::txthreshold){       
				para_tapa = OPTDENSE::para_tapa;  
			}
			else{
				if(tx < OPTDENSE::txthreshold2)         
				{
					para_tapa = OPTDENSE::para_tapa2;
				}
			}
		}    

		score = (1.f - para_tapa)*score_ncc + para_tapa*score_geo ;      
        score = (1.f - para_part)*score + para_part*score_gra ;
		score = (1.f - OPTDENSE::photometric_flow)*score_ncc + OPTDENSE::photometric_flow*score_flow ;

		
		
		
		

		
		float para_prior = OPTDENSE::para_prior ;
		float fsigmaPrior = OPTDENSE::fsigmaPrior ;

		

		if(!image0.depthMapPrior.empty())
		{		
			if (image0.depthMapPrior(x0) != 0)
			{
				const Depth depthDifference = DepthSimilarity(image0.depthMapPrior(x0),depth); 
				const float weightPrior = EXP(-SQUARE(depthDifference) / (2 * SQUARE(fsigmaPrior)));
				
				score = score * (1.f - para_prior) + 2 * (1.f - weightPrior) * para_prior ; 	
				 
				
			}
		}


		
	}
	

	ASSERT(ISFINITE(score));
    
	return score;
}


// compute pixel's NCC score
float DepthEstimator::ScorePixel(Depth depth, const Normal& normal)
{
	ASSERT(depth > 0 && normal.dot(Cast<float>(static_cast<const Point3&>(X0))) <= 0);
	// compute score for this pixel as seen in each view
	ASSERT(scores.size() == images.size());
	FOREACH(idxView, images)
		scores[idxView] = ScorePixelImage(idxView, images[idxView], depth, normal);
	#if DENSE_AGGNCC == DENSE_AGGNCC_NTH
	// set score as the nth element
	return scores.GetNth(idxScore);
	#elif DENSE_AGGNCC == DENSE_AGGNCC_MEAN
	// set score as the average similarity
	#if 1
	return scores.mean();
	#else
	const float* pscore(scores.data());
	const float* pescore(pscore+scores.rows());
	float score(0);
	do {
		score += MINF(*pscore, thRobust);
	} while (++pscore <= pescore);
	return score/scores.rows();
	#endif
	#elif DENSE_AGGNCC == DENSE_AGGNCC_MIN
	// set score as the min similarity
	return scores.minCoeff();
	#else
	// set score as the min-mean similarity
	if (idxScore == 0)
		return *std::min_element(scores.cbegin(), scores.cend());
	#if 0
	return std::accumulate(scores.cbegin(), &scores.GetNth(idxScore), 0.f) / idxScore;
	#elif 1
	const float* pescore(&scores.GetNth(idxScore));
	const float* pscore(scores.cbegin());
	int n(1); float score(*pscore);
	do {
		const float s(*(++pscore));
		if (s >= thRobust)
			break;
		score += s;
		++n;
	} while (pscore < pescore);
	return score/n;
	#else
	const float thScore(MAXF(*std::min_element(scores.cbegin(), scores.cend()), 0.05f)*2);
	const float* pscore(scores.cbegin());
	const float* pescore(pscore+scores.size());
	int n(0); float score(0);
	do {
		const float s(*pscore);
		if (s <= thScore) {
			score += s;
			++n;
		}
	} while (++pscore <= pescore);
	return score/n;
	#endif
	#endif
}

// run propagation and random refinement cycles;
// the solution belonging to the target image can be also propagated
void DepthEstimator::ProcessPixel(IDX idx)
{
	// compute pixel coordinates from pixel index and its neighbors
	ASSERT(dir == LT2RB || dir == RB2LT);
	if (!PreparePixelPatch(dir == LT2RB ? coords[idx] : coords[coords.GetSize()-1-idx]) || !FillPixelPatch())
		return;
	// find neighbors
	neighbors.Empty();
	#if DENSE_SMOOTHNESS != DENSE_SMOOTHNESS_NA
	neighborsClose.Empty();
	#endif
	candidate.Empty();	


	if(nIteration_external>= 1 ){	

		#if 1
		uint id_0 = image0.GetID(); 	  	   	 				  
		float tx(arrDepthData0[id_0].graMap(x0)); 

		int propagationHalfWindow = OPTDENSE::propagatehalfwin ;          
		int step = OPTDENSE::propagatestep ;						      
		if(tx > 150){  
			propagationHalfWindow = 5 ;			 
		}
		else{
			propagationHalfWindow = OPTDENSE::propagatehalfwin ;
		}
		#else
		int propagationHalfWindow = OPTDENSE::propagatehalfwin ;	
		int step = OPTDENSE::propagatestep ;	
		#endif

		// if (dir == LT2RB) {
		// 	if (x0.x > nSizeHalfWindow && x0.y > nSizeHalfWindow && x0.x < size.width-nSizeHalfWindow && x0.y < size.height-nSizeHalfWindow ){
		// 		const ImageRef candidate[8] = {
		// 			ImageRef(x0.x-1, x0.y),
		// 			// ImageRef(x0.x-3, x0.y),
		// 			ImageRef(x0.x-5, x0.y),
		// 			ImageRef(x0.x, x0.y-1),
		// 			// ImageRef(x0.x, x0.y-3),
		// 			ImageRef(x0.x, x0.y-5),
		// 			ImageRef(x0.x+1, x0.y),
		// 			// ImageRef(x0.x+3, x0.y),
		// 			ImageRef(x0.x+5, x0.y),
		// 			ImageRef(x0.x, x0.y+1),
		// 			// ImageRef(x0.x, x0.y+3),
		// 			ImageRef(x0.x, x0.y+5)
		// 			// ImageRef(x0.x-2, x0.y-1),
		// 			// ImageRef(x0.x+2, x0.y-1),
		// 			// ImageRef(x0.x-2, x0.y+1),
		// 			// ImageRef(x0.x+2, x0.y+1),
		// 			// ImageRef(x0.x-1, x0.y-2),
		// 			// ImageRef(x0.x-1, x0.y+2),
		// 			// ImageRef(x0.x+1, x0.y-2),
		// 			// ImageRef(x0.x+1, x0.y+2)
		// 		};
		// 		for (int n=0; n<8; ++n) {
		// 			const ImageRef& nx = candidate[n];
		// 			const Depth ndepth(depthMap0(nx));
		// 			if (ndepth > 0) {
		// 				#if DENSE_SMOOTHNESS != DENSE_SMOOTHNESS_NA
		// 				neighbors.push_back(nx);
		// 				neighborsClose.push_back(NeighborEstimate{ndepth,normalMap0(nx)
		// 				#if DENSE_SMOOTHNESS == DENSE_SMOOTHNESS_PLANE
		// 				, Cast<float>(image0.camera.TransformPointI2C(Point3(nx, ndepth)))
		// 				#endif
		// 				});
		// 				#else
		// 				neighbors.push_back(NeighborData{nx,ndepth,normalMap0(nx)});
		// 				#endif
		// 			}
		// 		}
		// 	}
		// }
		// else{
		// 	ASSERT(dir == RB2LT);
		// 	if (x0.x > nSizeHalfWindow && x0.y > nSizeHalfWindow && x0.x < size.width-nSizeHalfWindow && x0.y < size.height-nSizeHalfWindow ){ 
		// 		const ImageRef candidate[8] = {
		// 			ImageRef(x0.x-1, x0.y),
		// 			// ImageRef(x0.x-3, x0.y),
		// 			ImageRef(x0.x-5, x0.y),
		// 			ImageRef(x0.x, x0.yuint id_0 = image0.GetID();-1),
		// 			// ImageRef(x0.x, x0.y-3),
		// 			ImageRef(x0.x, x0.y-5),
		// 			ImageRef(x0.x+1, x0.y),
		// 			// ImageRef(x0.x+3, x0.y),
		// 			ImageRef(x0.x+5, x0.y),
		// 			ImageRef(x0.x, x0.y+1),
		// 			// ImageRef(x0.x, x0.y+3),
		// 			ImageRef(x0.x, x0.y+5)
		// 			// ImageRef(x0.x-2, x0.y-1),
		// 			// ImageRef(x0.x+2, x0.y-1),
		// 			// ImageRef(x0.x-2, x0.y+1),
		// 			// ImageRef(x0.x+2, x0.y+1),
		// 			// ImageRef(x0.x-1, x0.y-2),
		// 			// ImageRef(x0.x-1, x0.y+2),
		// 			// ImageRef(x0.x+1, x0.y-2),
		// 			// ImageRef(x0.x+1, x0.y+2)
		// 		};
		// 		for (int n=0; n<8; ++n) {
		// 			const ImageRef& nx = candidate[n];
		// 			const Depth ndepth(depthMap0(nx));
		// 			if (ndepth > 0) {
		// 				#if DENSE_SMOOTHNESS != DENSE_SMOOTHNESS_NA
		// 				neighbors.push_back(nx);
		// 				neighborsClose.push_back(NeighborEstimate{ndepth,normalMap0(nx)
		// 				#if DENSE_SMOOTHNESS == DENSE_SMOOTHNESS_PLANE
		// 				, Cast<float>(image0.camera.TransformPointI2C(Point3(nx, ndepth)))
		// 				#endif
		// 				});
		// 				#else
		// 				neighbors.push_back(NeighborData{nx,ndepth,normalMap0(nx)});
		// 				#endif
		// 			}
		// 		}
		// 	}
		// }
		
		
		
		if (dir == LT2RB) {
			
			if (x0.x > propagationHalfWindow && x0.y > propagationHalfWindow && x0.x < size.width-propagationHalfWindow && x0.y < size.height-propagationHalfWindow ){
				for (int i=1; i <= propagationHalfWindow; i+=step){
					candidate.push_back(Point2i(x0.x, x0.y - i)); 
					candidate.push_back(Point2i(x0.x, x0.y + i)); 
					candidate.push_back(Point2i(x0.x - i, x0.y)); 
					candidate.push_back(Point2i(x0.x + i, x0.y)); 
				}
			}
			else if(x0.x > nSizeHalfWindow && x0.y > nSizeHalfWindow && x0.x < size.width-nSizeHalfWindow && x0.y < size.height-nSizeHalfWindow){
				candidate.push_back(Point2i(x0.x, x0.y - 1)); 
				candidate.push_back(Point2i(x0.x, x0.y + 1)); 
				candidate.push_back(Point2i(x0.x - 1, x0.y)); 
				candidate.push_back(Point2i(x0.x + 1, x0.y)); 

			}
			FOREACH(n, candidate){
				const ImageRef& nx = candidate[n];
				const Depth ndepth(depthMap0(nx));
				if (ndepth > 0) {
					#if DENSE_SMOOTHNESS != DENSE_SMOOTHNESS_NA
					neighbors.push_back(nx);
					neighborsClose.push_back(NeighborEstimate{ndepth,normalMap0(nx)
					#if DENSE_SMOOTHNESS == DENSE_SMOOTHNESS_PLANE
					, Cast<float>(image0.camera.TransformPointI2C(Point3(nx, ndepth)))
					#endif
					});
					#else
					neighbors.push_back(NeighborData{nx,ndepth,normalMap0(nx)});
					#endif
				}
			}
		}
		else{
			ASSERT(dir == RB2LT);
			
			if (x0.x > propagationHalfWindow && x0.y > propagationHalfWindow && x0.x < size.width-propagationHalfWindow && x0.y < size.height-propagationHalfWindow ){
				for (int i=1; i <= propagationHalfWindow; i+=step){
					candidate.push_back(Point2i(x0.x, x0.y - i));
					candidate.push_back(Point2i(x0.x, x0.y + i)); 
					candidate.push_back(Point2i(x0.x - i, x0.y));
					candidate.push_back(Point2i(x0.x + i, x0.y)); 
				}
			}
			else if(x0.x > nSizeHalfWindow && x0.y > nSizeHalfWindow && x0.x < size.width-nSizeHalfWindow && x0.y < size.height-nSizeHalfWindow){
				candidate.push_back(Point2i(x0.x, x0.y - 1)); 
				candidate.push_back(Point2i(x0.x, x0.y + 1)); 
				candidate.push_back(Point2i(x0.x - 1, x0.y)); 
				candidate.push_back(Point2i(x0.x + 1, x0.y)); 
			}
			FOREACH(n, candidate){
				const ImageRef& nx = candidate[n];
				const Depth ndepth(depthMap0(nx));
				if (ndepth > 0) {
					#if DENSE_SMOOTHNESS != DENSE_SMOOTHNESS_NA
					neighbors.push_back(nx);
					neighborsClose.push_back(NeighborEstimate{ndepth,normalMap0(nx)
					#if DENSE_SMOOTHNESS == DENSE_SMOOTHNESS_PLANE
					, Cast<float>(image0.camera.TransformPointI2C(Point3(nx, ndepth)))
					#endif
					});
					#else
					neighbors.push_back(NeighborData{nx,ndepth,normalMap0(nx)});
					#endif
				}
			}
		}
	}
	else{

		if (dir == LT2RB) {
			// direction from left-top to right-bottom corner
			if (x0.x > nSizeHalfWindow) {
				const ImageRef nx(x0.x-1, x0.y);
				const Depth ndepth(depthMap0(nx));
				if (ndepth > 0) {
					#if DENSE_SMOOTHNESS != DENSE_SMOOTHNESS_NA
					neighbors.emplace_back(nx);
					neighborsClose.emplace_back(NeighborEstimate{ndepth,normalMap0(nx)
						#if DENSE_SMOOTHNESS == DENSE_SMOOTHNESS_PLANE
						, Cast<float>(image0.camera.TransformPointI2C(Point3(nx, ndepth)))
						#endif
					});
					#else
					neighbors.emplace_back(NeighborData{nx,ndepth,normalMap0(nx)});
					#endif
				}
			}
			if (x0.y > nSizeHalfWindow) {
				const ImageRef nx(x0.x, x0.y-1);
				const Depth ndepth(depthMap0(nx));
				if (ndepth > 0) {
					#if DENSE_SMOOTHNESS != DENSE_SMOOTHNESS_NA
					neighbors.emplace_back(nx);
					neighborsClose.emplace_back(NeighborEstimate{ndepth,normalMap0(nx)
						#if DENSE_SMOOTHNESS == DENSE_SMOOTHNESS_PLANE
						, Cast<float>(image0.camera.TransformPointI2C(Point3(nx, ndepth)))
						#endif
					});
					#else
					neighbors.emplace_back(NeighborData{nx,ndepth,normalMap0(nx)});
					#endif
				}
			}
			#if DENSE_SMOOTHNESS != DENSE_SMOOTHNESS_NA
			if (x0.x < size.width-nSizeHalfWindow) {
				const ImageRef nx(x0.x+1, x0.y);
				const Depth ndepth(depthMap0(nx));
				if (ndepth > 0)
					neighborsClose.emplace_back(NeighborEstimate{ndepth,normalMap0(nx)
						#if DENSE_SMOOTHNESS == DENSE_SMOOTHNESS_PLANE
						, Cast<float>(image0.camera.TransformPointI2C(Point3(nx, ndepth)))
						#endif
					});
			}
			if (x0.y < size.height-nSizeHalfWindow) {
				const ImageRef nx(x0.x, x0.y+1);
				const Depth ndepth(depthMap0(nx));
				if (ndepth > 0)
					neighborsClose.emplace_back(NeighborEstimate{ndepth,normalMap0(nx)
						#if DENSE_SMOOTHNESS == DENSE_SMOOTHNESS_PLANE
						, Cast<float>(image0.camera.TransformPointI2C(Point3(nx, ndepth)))
						#endif
					});
			}
			#endif
		} else {
			ASSERT(dir == RB2LT);
			// direction from right-bottom to left-top corner
			if (x0.x < size.width-nSizeHalfWindow) {
				const ImageRef nx(x0.x+1, x0.y);
				const Depth ndepth(depthMap0(nx));
				if (ndepth > 0) {
					#if DENSE_SMOOTHNESS != DENSE_SMOOTHNESS_NA
					neighbors.emplace_back(nx);
					neighborsClose.emplace_back(NeighborEstimate{ndepth,normalMap0(nx)
						#if DENSE_SMOOTHNESS == DENSE_SMOOTHNESS_PLANE
						, Cast<float>(image0.camera.TransformPointI2C(Point3(nx, ndepth)))
						#endif
					});
					#else
					neighbors.emplace_back(NeighborData{nx,ndepth,normalMap0(nx)});
					#endif
				}
			}
			if (x0.y < size.height-nSizeHalfWindow) {
				const ImageRef nx(x0.x, x0.y+1);
				const Depth ndepth(depthMap0(nx));
				if (ndepth > 0) {
					#if DENSE_SMOOTHNESS != DENSE_SMOOTHNESS_NA
					neighbors.emplace_back(nx);
					neighborsClose.emplace_back(NeighborEstimate{ndepth,normalMap0(nx)
						#if DENSE_SMOOTHNESS == DENSE_SMOOTHNESS_PLANE
						, Cast<float>(image0.camera.TransformPointI2C(Point3(nx, ndepth)))
						#endif
					});
					#else
					neighbors.emplace_back(NeighborData{nx,ndepth,normalMap0(nx)});
					#endif
				}
			}
			#if DENSE_SMOOTHNESS != DENSE_SMOOTHNESS_NA
			if (x0.x > nSizeHalfWindow) {
				const ImageRef nx(x0.x-1, x0.y);
				const Depth ndepth(depthMap0(nx));
				if (ndepth > 0)
					neighborsClose.emplace_back(NeighborEstimate{ndepth,normalMap0(nx)
						#if DENSE_SMOOTHNESS == DENSE_SMOOTHNESS_PLANE
						, Cast<float>(image0.camera.TransformPointI2C(Point3(nx, ndepth)))
						#endif
					});
			}
			if (x0.y > nSizeHalfWindow) {
				const ImageRef nx(x0.x, x0.y-1);
				const Depth ndepth(depthMap0(nx));
				if (ndepth > 0)
					neighborsClose.emplace_back(NeighborEstimate{ndepth,normalMap0(nx)
						#if DENSE_SMOOTHNESS == DENSE_SMOOTHNESS_PLANE
						, Cast<float>(image0.camera.TransformPointI2C(Point3(nx, ndepth)))
						#endif
					});
			}
			#endif
		}
	}



	
	uint id_0 = image0.GetID();		
	DepthData& depthData_image0 = arrDepthData0[id_0] ;
	float& conf = confMap0(x0);
	Depth& depth = depthMap0(x0);
	Normal& normal = normalMap0(x0);
	const Normal viewDir(Cast<float>(reinterpret_cast<const Point3&>(X0)));
	ASSERT(depth > 0 && normal.dot(viewDir) <= 0);
	#if DENSE_REFINE == DENSE_REFINE_ITER
	// check if any of the neighbor estimates are better then the current estimate
	#if DENSE_SMOOTHNESS != DENSE_SMOOTHNESS_NA
	FOREACH(n, neighbors) {
		const ImageRef& nx = neighbors[n];
	#else
	for (NeighborData& neighbor: neighbors) {
		const ImageRef& nx = neighbor.x;
	#endif
		if (confMap0(nx) >= OPTDENSE::fNCCThresholdKeep)
			continue;
		#if DENSE_SMOOTHNESS != DENSE_SMOOTHNESS_NA
		NeighborEstimate& neighbor = neighborsClose[n];
		#endif
		neighbor.depth = InterpolatePixel(nx, neighbor.depth, neighbor.normal);
		CorrectNormal(neighbor.normal);
		ASSERT(neighbor.depth > 0 && neighbor.normal.dot(viewDir) <= 0);
		#if DENSE_SMOOTHNESS == DENSE_SMOOTHNESS_PLANE
		InitPlane(neighbor.depth, neighbor.normal);
		#endif
		const float nconf(ScorePixel(neighbor.depth, neighbor.normal));
		ASSERT(nconf >= 0 && nconf <= 2);
		if (conf > nconf) {
			conf = nconf;
			depth = neighbor.depth;
			normal = neighbor.normal;
			if(nIteration_external<OPTDENSE::photo2geo)
			{
				REAL c2p_cur(normal.dot(image0.camera.TransformPointI2C(Point3(x0, depth)))); 
				if(c2p_cur < 0)
				c2p_cur = -1.f*c2p_cur;
				if(c2p_cur <= 0)
				std::cout<<"c2p_cur error"<<std::endl;
				if(c2p_cur > delta_c2pmax)
				delta_c2pmax = c2p_cur;
			}
		}
	}
	// try random values around the current estimate in order to refine it
	unsigned idxScaleRange(0);
	RefineIters:
	if (conf <= thConfSmall)
		idxScaleRange = 2;
	else if (conf <= thConfBig)
		idxScaleRange = 1;
	else if (conf >= thConfRand) {
		// try completely random values in order to find an initial estimate
		for (unsigned iter=0; iter<OPTDENSE::nRandomIters; ++iter) {
			const Depth ndepth(RandomDepth(dMinSqr, dMaxSqr));
			const Normal nnormal(RandomNormal(viewDir));
			const float nconf(ScorePixel(ndepth, nnormal));
			ASSERT(nconf >= 0);
			if (conf > nconf) {
				conf = nconf;
				depth = ndepth;
				normal = nnormal;
				
				if (conf < thConfRand)
					goto RefineIters;
			}
		}
		return;
	}
	float scaleRange(scaleRanges[idxScaleRange]);
	const float depthRange(MaxDepthDifference(depth, OPTDENSE::fRandomDepthRatio));
	Point2f p;
	Normal2Dir(normal, p);
	Normal nnormal;
	for (unsigned iter=0; iter<OPTDENSE::nRandomIters; ++iter) {
		const Depth ndepth(rnd.randomMeanRange(depth, depthRange*scaleRange));
		if (!ISINSIDE(ndepth, dMin, dMax))
			continue;
		const Point2f np(rnd.randomMeanRange(p.x, angle1Range*scaleRange), rnd.randomMeanRange(p.y, angle2Range*scaleRange));
		Dir2Normal(np, nnormal);
		if (nnormal.dot(viewDir) >= 0)
			continue;
		#if DENSE_SMOOTHNESS == DENSE_SMOOTHNESS_PLANE
		InitPlane(ndepth, nnormal);
		#endif
		const float nconf(ScorePixel(ndepth, nnormal));
		ASSERT(nconf >= 0);
		if (conf > nconf) {
			conf = nconf;
			depth = ndepth;
			normal = nnormal;
			p = np;
			scaleRange = scaleRanges[++idxScaleRange];
			// if(nIteration_external<OPTDENSE::photo2geo)
			// {
			// 	REAL c2p_cur(normal.dot(image0.camera.TransformPointI2C(Point3(x0, depth))));
			// 	if(c2p_cur < 0)
			// 	c2p_cur = -1.f*c2p_cur;
			// 	if(c2p_cur <= 0)
			// 	std::cout<<"c2p_cur error"<<std::endl;
			// 	if(c2p_cur > delta_c2pmax)
			// 	delta_c2pmax = c2p_cur;
			// }
		}
	}

	if(OPTDENSE::viewspread){
		if(nIteration_external>= 1 )
		{	
			FOREACH(idxView, images)
			{
				if(idxView == 0)
				continue;
				// std::cout<<idxView<<std::endl;
				uint id_1 = images[idxView].view.GetID(); 
				DepthData& depthData_image1 = arrDepthData0[id_1] ;
				const ViewData& image1 = images[idxView];
				Matrix3x3f H(ComputeHomographyMatrix(image1 ,depth ,normal));
				Point3f X1;
				ProjectVertex_3x3_2_3(H.val , Point2f(float(x0.x),float(x0.y)).ptr() , X1.ptr());
				const Point2f x1_f(X1) ; 
				Point2i x1_i; 			 
				x1_i.x = (int)(x1_f.x) ;
				x1_i.y = (int)(x1_f.y) ;

				neighbors.Empty();
				#if DENSE_SMOOTHNESS != DENSE_SMOOTHNESS_NA
				neighborsClose.Empty();
				#endif
				candidate.Empty();
				
				
				if (dir == LT2RB) {
					if(x1_i.x > nSizeHalfWindow && x1_i.y > nSizeHalfWindow && x1_i.x < size.width-nSizeHalfWindow && x1_i.y < size.height-nSizeHalfWindow){
						candidate.push_back(Point2i(x1_i.x, x1_i.y - 1)); 
						candidate.push_back(Point2i(x1_i.x, x1_i.y + 1)); 
						candidate.push_back(Point2i(x1_i.x - 1, x1_i.y)); 
						candidate.push_back(Point2i(x1_i.x + 1, x1_i.y)); 

					}
					FOREACH(n, candidate){
						const ImageRef& nx = candidate[n];
						const Depth ndepth(depthData_image1.depthMap(nx));
						if (ndepth > 0) {
							#if DENSE_SMOOTHNESS != DENSE_SMOOTHNESS_NA
							neighbors.push_back(nx);
							neighborsClose.push_back(NeighborEstimate{ndepth,depthData_image1.normalMap(nx)
							#if DENSE_SMOOTHNESS == DENSE_SMOOTHNESS_PLANE
							, Cast<float>(depthData_image1.images[1].camera.TransformPointI2C(Point3(nx, ndepth)))
							#endif
							});
							#else
							neighbors.push_back(NeighborData{nx,ndepth,normalMap0(nx)});
							#endif
						}
					}
				}
				else{
					ASSERT(dir == RB2LT);
					if(x1_i.x > nSizeHalfWindow && x1_i.y > nSizeHalfWindow && x1_i.x < size.width-nSizeHalfWindow && x1_i.y < size.height-nSizeHalfWindow){
						candidate.push_back(Point2i(x1_i.x, x1_i.y - 1)); 
						candidate.push_back(Point2i(x1_i.x, x1_i.y + 1));
						candidate.push_back(Point2i(x1_i.x - 1, x1_i.y)); 
						candidate.push_back(Point2i(x1_i.x + 1, x1_i.y)); 
					
					FOREACH(n, candidate){
						const ImageRef& nx = candidate[n];
						const Depth ndepth(depthData_image1.depthMap(nx));
						if (ndepth > 0) {
							#if DENSE_SMOOTHNESS != DENSE_SMOOTHNESS_NA
							neighbors.push_back(nx);
							neighborsClose.push_back(NeighborEstimate{ndepth,depthData_image1.normalMap(nx)
							#if DENSE_SMOOTHNESS == DENSE_SMOOTHNESS_PLANE
							, Cast<float>(depthData_image1.images[0].camera.TransformPointI2C(Point3(nx, ndepth)))
							#endif
							});
							#else
							neighbors.push_back(NeighborData{nx,ndepth,normalMap0(nx)});
							#endif
						}
					}
				}
			
		
				FOREACH(n, neighbors) {
					const ImageRef& nx = neighbors[n];
					if (depthData_image1.confMap(nx) >= OPTDENSE::fNCCThresholdKeep)
						continue;
					#if DENSE_SMOOTHNESS != DENSE_SMOOTHNESS_NA
					NeighborEstimate& neighbor = neighborsClose[n];
					#endif
					const Point3f X(depthData_image1.images[0].camera.TransformPointI2W(Point3(nx.x,nx.y, neighbor.depth)));
					const Point3f X0(depthData_image0.images[0].camera.TransformPointW2C(Point3(X.x,X.y,X.z)));
					neighbor.depth = X0.z;
					CorrectNormal(neighbor.normal);
					ASSERT(neighbor.depth > 0 && neighbor.normal.dot(viewDir) <= 0);
					#if DENSE_SMOOTHNESS == DENSE_SMOOTHNESS_PLANE
					InitPlane(neighbor.depth, neighbor.normal);
					#endif
					const float nconf(ScorePixel(neighbor.depth, neighbor.normal));
					ASSERT(nconf >= 0 && nconf <= 2);
					if (conf > nconf) {
						conf = nconf;
						depth = neighbor.depth;
						normal = neighbor.normal;
					}
				}
			}
		}
	}

	if (nIteration_external == OPTDENSE::nEstimationIters_external - 1 && nIteration == OPTDENSE::nEstimationIters - 1 )
		{
			
			Depth& mdepth =  depthData_image0.meanshiftpriors_depthMap(x0);
			Normal& mnormal = depthData_image0.meanshiftpriors_normalMap(x0);
			Depth& ndepth = depthData_image0.nresize_depthMap(x0);
			Normal& nnormal = depthData_image0.nresize_normalMap(x0);
			const Normal viewDir(Cast<float>(reinterpret_cast<const Point3&>(X0)));
			ASSERT(ndepth > 0 && nnormal.dot(viewDir) <= 0);
			ndepth = InterpolatePixel(x0, ndepth, nnormal);
			CorrectNormal(nnormal);
			ASSERT(ndepth > 0 && nnormal.dot(viewDir) <= 0);
			InitPlane(ndepth, nnormal);
			const float nconf(ScorePixel(ndepth, nnormal));
			ASSERT(nconf >= 0);
			if (conf > nconf - 0.1f) {
				conf = nconf;
				depth = ndepth;
				normal = nnormal;

			}
			
		}


	#else
	// current pixel estimate
	PixelEstimate currEstimate{depth, normal};
	// propagate depth estimate from the best neighbor estimate
	PixelEstimate prevEstimate; float prevCost(FLT_MAX);
	#if DENSE_SMOOTHNESS != DENSE_SMOOTHNESS_NA
	FOREACH(n, neighbors) {
		const ImageRef& nx = neighbors[n];
	#else
	for (const NeighborData& neighbor: neighbors) {
		const ImageRef& nx = neighbor.x;
	#endif
		float nconf(confMap0(nx));
		const unsigned nidxScaleRange(DecodeScoreScale(nconf));
		ASSERT(nconf >= 0 && nconf <= 2);
		if (nconf >= OPTDENSE::fNCCThresholdKeep)
			continue;
		if (prevCost <= nconf)
			continue;
		#if DENSE_SMOOTHNESS != DENSE_SMOOTHNESS_NA
		const NeighborEstimate& neighbor = neighborsClose[n];
		#endif
		if (neighbor.normal.dot(viewDir) >= 0)
			continue;
		prevEstimate.depth = InterpolatePixel(nx, neighbor.depth, neighbor.normal);
		prevEstimate.normal = neighbor.normal;
		CorrectNormal(prevEstimate.normal);
		prevCost = nconf;
	}
	if (prevCost == FLT_MAX)
		prevEstimate = PerturbEstimate(currEstimate, thPerturbation);
	// randomly sampled estimate
	PixelEstimate randEstimate(PerturbEstimate(currEstimate, thPerturbation));
	// select best pixel estimate
	const int numCosts = 5;
	float costs[numCosts] = {0,0,0,0,0};
	const Depth depths[numCosts] = {
		currEstimate.depth, prevEstimate.depth, randEstimate.depth,
		currEstimate.depth, randEstimate.depth};
	const Normal normals[numCosts] = {
		currEstimate.normal, prevEstimate.normal,
		randEstimate.normal, randEstimate.normal,
		currEstimate.normal};
	conf = FLT_MAX;
	for (int idxCost=0; idxCost<numCosts; ++idxCost) {
		const Depth ndepth(depths[idxCost]);
		const Normal nnormal(normals[idxCost]);
		#if DENSE_SMOOTHNESS == DENSE_SMOOTHNESS_PLANE
		InitPlane(ndepth, nnormal);
		#endif
		const float nconf(ScorePixel(ndepth, nnormal));
		ASSERT(nconf >= 0);
		if (conf > nconf) {
			conf = nconf;
			depth = ndepth;
			normal = nnormal;
		}
	}
	#endif
}

// interpolate given pixel's estimate to the current position
Depth DepthEstimator::InterpolatePixel(const ImageRef& nx, Depth depth, const Normal& normal) const
{
	ASSERT(depth > 0 && normal.dot(image0.camera.TransformPointI2C(Cast<REAL>(nx))) <= 0);
	Depth depthNew;
	#if 0 	
	// compute as intersection of the lines
	// {(x1, y1), (x2, y2)} from neighbor's 3D point towards normal direction
	// and
	// {(0, 0), (x4, 1)} from camera center towards current pixel direction
	// in the x or y plane
	if (x0.x == nx.x) {
		const float fy = (float)image0.camera.K[4];
		const float cy = (float)image0.camera.K[5];
		const float x1 = depth * (nx.y - cy) / fy;
		const float y1 = depth;
		const float x4 = (x0.y - cy) / fy;
		const float denom = normal.z + x4 * normal.y;
		if (ISZERO(denom))
			return depth;
		const float x2 = x1 + normal.z;
		const float y2 = y1 - normal.y;
		const float nom = y1 * x2 - x1 * y2;
		depthNew = nom / denom;
	}
	else {
		ASSERT(x0.y == nx.y);
		const float fx = (float)image0.camera.K[0];
		const float cx = (float)image0.camera.K[2];
		ASSERT(image0.camera.K[1] == 0);
		const float x1 = depth * (nx.x - cx) / fx;
		const float y1 = depth;
		const float x4 = (x0.x - cx) / fx;
		const float denom = normal.z + x4 * normal.x;
		if (ISZERO(denom))
			return depth;
		const float x2 = x1 + normal.z;
		const float y2 = y1 - normal.x;
		const float nom = y1 * x2 - x1 * y2;
		depthNew = nom / denom;
	}
	#else
	// compute as the ray - plane intersection
	{
		#if 0
		const Plane plane(Cast<REAL>(normal), image0.camera.TransformPointI2C(Point3(nx, depth)));
		const Ray3 ray(Point3::ZERO, normalized(X0));
		depthNew = (Depth)ray.Intersects(plane).z();
		#else
		const Point3 planeN(normal);
		const REAL planeD(planeN.dot(image0.camera.TransformPointI2C(Point3(nx, depth))));
		depthNew = (Depth)(planeD / planeN.dot(reinterpret_cast<const Point3&>(X0)));
		#endif
	}
	#endif
	return ISINSIDE(depthNew,dMin,dMax) ? depthNew : depth;
}

#if DENSE_SMOOTHNESS == DENSE_SMOOTHNESS_PLANE
// compute plane defined by current depth and normal estimate
void DepthEstimator::InitPlane(Depth depth, const Normal& normal)
{
	#if 0
	plane.Set(reinterpret_cast<const Vec3f&>(normal), Vec3f(depth*Cast<float>(X0)));
	#else
	plane.m_vN = reinterpret_cast<const Vec3f&>(normal);
	plane.m_fD = -depth*reinterpret_cast<const Vec3f&>(normal).dot(Cast<float>(X0));
	#endif
}
#endif

#if DENSE_REFINE == DENSE_REFINE_EXACT
DepthEstimator::PixelEstimate DepthEstimator::PerturbEstimate(const PixelEstimate& est, float perturbation)
{
	PixelEstimate ptbEst;

	// perturb depth
	const float minDepth = est.depth * (1.f-perturbation);
	const float maxDepth = est.depth * (1.f+perturbation);
	ptbEst.depth = CLAMP(rnd.randomUniform(minDepth, maxDepth), dMin, dMax);

	// perturb normal
	const Normal viewDir(Cast<float>(static_cast<const Point3&>(X0)));
	std::uniform_real_distribution<float> urd(-1.f, 1.f);
	const int numMaxTrials = 3;
	int numTrials = 0;
	perturbation *= FHALF_PI;
	while(true) {
		// generate random perturbation rotation
		const RMatrixBaseF R(urd(rnd)*perturbation, urd(rnd)*perturbation, urd(rnd)*perturbation);
		// perturb normal vector
		ptbEst.normal = R * est.normal;
		// make sure the perturbed normal is still looking towards the camera,
		// otherwise try again with a smaller perturbation
		if (ptbEst.normal.dot(viewDir) < 0.f)
			break;
		if (++numTrials == numMaxTrials) {
			ptbEst.normal = est.normal;
			return ptbEst;
		}
		perturbation *= 0.5f;
	}
	ASSERT(ISEQUAL(norm(ptbEst.normal), 1.f));

	return ptbEst;
}
#endif
/*----------------------------------------------------------------*/



// S T R U C T S ///////////////////////////////////////////////////

namespace CGAL {
typedef CGAL::Simple_cartesian<double> kernel_t;
typedef CGAL::Projection_traits_xy_3<kernel_t> Geometry;
typedef CGAL::Delaunay_triangulation_2<Geometry> Delaunay;
typedef CGAL::Delaunay::Face_circulator FaceCirculator;
typedef CGAL::Delaunay::Face_handle FaceHandle;
typedef CGAL::Delaunay::Vertex_circulator VertexCirculator;
typedef CGAL::Delaunay::Vertex_handle VertexHandle;
typedef kernel_t::Point_3 Point;
}

// triangulate in-view points, generating a 2D mesh
// return also the estimated depth boundaries (min and max depth)
std::pair<float,float> TriangulatePointsDelaunay(const DepthData::ViewData& image, const PointCloud& pointcloud, const IndexArr& points, CGAL::Delaunay& delaunay)
{
	ASSERT(sizeof(Point3) == sizeof(X3D));
	ASSERT(sizeof(Point3) == sizeof(CGAL::Point));
	std::pair<float,float> depthBounds(FLT_MAX, 0.f);
	for (uint32_t idx: points) {
		const Point3f pt(image.camera.ProjectPointP3(pointcloud.points[idx]));
		delaunay.insert(CGAL::Point(pt.x/pt.z, pt.y/pt.z, pt.z));
		if (depthBounds.first > pt.z)
			depthBounds.first = pt.z;
		if (depthBounds.second < pt.z)
			depthBounds.second = pt.z;
	}
	// if full size depth-map requested
	if (OPTDENSE::bAddCorners) {
		typedef TIndexScore<float,float> DepthDist;
		typedef CLISTDEF0(DepthDist) DepthDistArr;
		typedef Eigen::Map< Eigen::VectorXf, Eigen::Unaligned, Eigen::InnerStride<2> > FloatMap;
		// add the four image corners at the average depth
		ASSERT(image.pImageData->IsValid() && ISINSIDE(image.pImageData->avgDepth, depthBounds.first, depthBounds.second));
		const CGAL::VertexHandle vcorners[] = {
			delaunay.insert(CGAL::Point(0, 0, image.pImageData->avgDepth)),
			delaunay.insert(CGAL::Point(image.image.width(), 0, image.pImageData->avgDepth)),
			delaunay.insert(CGAL::Point(0, image.image.height(), image.pImageData->avgDepth)),
			delaunay.insert(CGAL::Point(image.image.width(), image.image.height(), image.pImageData->avgDepth))
		};
		// compute average depth from the closest 3 directly connected faces,
		// weighted by the distance
		const size_t numPoints = 3;
		for (int i=0; i<4; ++i) {
			const CGAL::VertexHandle vcorner = vcorners[i];
			CGAL::FaceCirculator cfc(delaunay.incident_faces(vcorner));
			if (cfc == 0)
				continue; // normally this should never happen
			const CGAL::FaceCirculator done(cfc);
			Point3d& poszA = reinterpret_cast<Point3d&>(vcorner->point());
			const Point2d& posA = reinterpret_cast<const Point2d&>(poszA);
			const Ray3d rayA(Point3d::ZERO, normalized(image.camera.TransformPointI2C(poszA)));
			DepthDistArr depths(0, numPoints);
			do {
				CGAL::FaceHandle fc(cfc->neighbor(cfc->index(vcorner)));
				if (fc == delaunay.infinite_face())
					continue;
				for (int j=0; j<4; ++j)
					if (fc->has_vertex(vcorners[j]))
						goto Continue;
				// compute the depth as the intersection of the corner ray with
				// the plane defined by the face's vertices
				{
				const Point3d& poszB0 = reinterpret_cast<const Point3d&>(fc->vertex(0)->point());
				const Point3d& poszB1 = reinterpret_cast<const Point3d&>(fc->vertex(1)->point());
				const Point3d& poszB2 = reinterpret_cast<const Point3d&>(fc->vertex(2)->point());
				const Planed planeB(
					image.camera.TransformPointI2C(poszB0),
					image.camera.TransformPointI2C(poszB1),
					image.camera.TransformPointI2C(poszB2)
				);
				const Point3d poszB(rayA.Intersects(planeB));
				if (poszB.z <= 0)
					continue;
				const Point2d posB((
					reinterpret_cast<const Point2d&>(poszB0)+
					reinterpret_cast<const Point2d&>(poszB1)+
					reinterpret_cast<const Point2d&>(poszB2))/3.f
				);
				const double dist(norm(posB-posA));
				depths.StoreTop<numPoints>(DepthDist(CLAMP((float)poszB.z,depthBounds.first,depthBounds.second), INVERT((float)dist)));
				}
				Continue:;
			} while (++cfc != done);
			if (depths.size() != numPoints)
				continue; // normally this should never happen
			FloatMap vecDists(&depths[0].score, numPoints);
			vecDists *= 1.f/vecDists.sum();
			FloatMap vecDepths(&depths[0].idx, numPoints);
			poszA.z = vecDepths.dot(vecDists);
		}
	}
	return depthBounds;
}

// roughly estimate depth and normal maps by triangulating the sparse point cloud
// and interpolating normal and depth for all pixels
bool MVS::TriangulatePoints2DepthMap(
	const DepthData::ViewData& image, const PointCloud& pointcloud, const IndexArr& points,
	DepthMap& depthMap, NormalMap& normalMap, Depth& dMin, Depth& dMax)
{
	ASSERT(image.pImageData != NULL);

	// triangulate in-view points
	CGAL::Delaunay delaunay;
	const std::pair<float,float> thDepth(TriangulatePointsDelaunay(image, pointcloud, points, delaunay));
	dMin = thDepth.first;
	dMax = thDepth.second;

	// create rough depth-map by interpolating inside triangles
	const Camera& camera = image.camera;
	depthMap.create(image.image.size());
	normalMap.create(image.image.size());
	if (!OPTDENSE::bAddCorners) {
		depthMap.memset(0);
		normalMap.memset(0);
	}
	struct RasterDepthDataPlaneData {
		const Camera& P;
		DepthMap& depthMap;
		NormalMap& normalMap;
		Point3f normal;
		Point3f normalPlane;
		inline void operator()(const ImageRef& pt) {
			if (!depthMap.isInside(pt))
				return;
			const Depth z(INVERT(normalPlane.dot(P.TransformPointI2C(Point2f(pt)))));
			if (z <= 0) // due to numerical instability
				return;
			depthMap(pt) = z;
			normalMap(pt) = normal;
		}
	};
	RasterDepthDataPlaneData data = {camera, depthMap, normalMap};
	for (CGAL::Delaunay::Face_iterator it=delaunay.faces_begin(); it!=delaunay.faces_end(); ++it) {
		const CGAL::Delaunay::Face& face = *it;
		const Point3f i0(reinterpret_cast<const Point3d&>(face.vertex(0)->point()));
		const Point3f i1(reinterpret_cast<const Point3d&>(face.vertex(1)->point()));
		const Point3f i2(reinterpret_cast<const Point3d&>(face.vertex(2)->point()));
		// compute the plane defined by the 3 points
		const Point3f c0(camera.TransformPointI2C(i0));
		const Point3f c1(camera.TransformPointI2C(i1));
		const Point3f c2(camera.TransformPointI2C(i2));
		const Point3f edge1(c1-c0);
		const Point3f edge2(c2-c0);
		data.normal = normalized(edge2.cross(edge1));
		data.normalPlane = data.normal * INVERT(data.normal.dot(c0));
		// draw triangle and for each pixel compute depth as the ray intersection with the plane
		Image8U::RasterizeTriangle(
			reinterpret_cast<const Point2f&>(i2),
			reinterpret_cast<const Point2f&>(i1),
			reinterpret_cast<const Point2f&>(i0), data);
	}
	return true;
} // TriangulatePoints2DepthMap
// same as above, but does not estimate the normal-map
bool MVS::TriangulatePoints2DepthMap(
	const DepthData::ViewData& image, const PointCloud& pointcloud, const IndexArr& points,
	DepthMap& depthMap, Depth& dMin, Depth& dMax)
{
	ASSERT(image.pImageData != NULL);

	// triangulate in-view points
	CGAL::Delaunay delaunay;
	const std::pair<float,float> thDepth(TriangulatePointsDelaunay(image, pointcloud, points, delaunay));
	dMin = thDepth.first;
	dMax = thDepth.second;

	// create rough depth-map by interpolating inside triangles
	const Camera& camera = image.camera;
	depthMap.create(image.image.size());
	if (!OPTDENSE::bAddCorners)
		depthMap.memset(0);
	struct RasterDepthDataPlaneData {
		const Camera& P;
		DepthMap& depthMap;
		Point3f normalPlane;
		inline void operator()(const ImageRef& pt) {
			if (!depthMap.isInside(pt))
				return;
			const Depth z((Depth)INVERT(normalPlane.dot(P.TransformPointI2C(Point2f(pt)))));
			if (z <= 0) // due to numerical instability
				return;
			depthMap(pt) = z;
		}
	};
	RasterDepthDataPlaneData data = {camera, depthMap};
	for (CGAL::Delaunay::Face_iterator it=delaunay.faces_begin(); it!=delaunay.faces_end(); ++it) {
		const CGAL::Delaunay::Face& face = *it;
		const Point3f i0(reinterpret_cast<const Point3d&>(face.vertex(0)->point()));
		const Point3f i1(reinterpret_cast<const Point3d&>(face.vertex(1)->point()));
		const Point3f i2(reinterpret_cast<const Point3d&>(face.vertex(2)->point()));
		// compute the plane defined by the 3 points
		const Point3f c0(camera.TransformPointI2C(i0));
		const Point3f c1(camera.TransformPointI2C(i1));
		const Point3f c2(camera.TransformPointI2C(i2));
		const Point3f edge1(c1-c0);
		const Point3f edge2(c2-c0);
		const Normal normal(normalized(edge2.cross(edge1)));
		data.normalPlane = normal * INVERT(normal.dot(c0));
		// draw triangle and for each pixel compute depth as the ray intersection with the plane
		Image8U::RasterizeTriangle(
			reinterpret_cast<const Point2f&>(i2),
			reinterpret_cast<const Point2f&>(i1),
			reinterpret_cast<const Point2f&>(i0), data);
	}
	return true;
} // TriangulatePoints2DepthMap
/*----------------------------------------------------------------*/


namespace MVS {

class PlaneSolverAdaptor
{
public:
	enum { MINIMUM_SAMPLES = 3 };
	enum { MAX_MODELS = 1 };

	typedef Plane Model;
	typedef cList<Model> Models;

	PlaneSolverAdaptor(const Point3Arr& points)
		: points_(points)
	{
	}
	PlaneSolverAdaptor(const Point3Arr& points, float w, float h, float d)
		: points_(points)
	{
		// LogAlpha0 is used to make error data scale invariant
		// Ratio of containing diagonal image rectangle over image area
		const float D = SQRT(w*w + h*h + d*d); // diameter
		const float A = w*h*d+1.f; // volume
		logalpha0_ = LOG10(2.0f*D/A*0.5f);
	}

	inline bool Fit(const std::vector<size_t>& samples, Models& models) const {
		Point3 points[3];
		for (size_t i=0; i<samples.size(); ++i)
			points[i] = points_[samples[i]];
		if (CheckCollinearity(points, 3))
			return false;
		models.Resize(1);
		models[0] = Plane(points[0], points[1], points[2]);
		return true;
	}

	inline void EvaluateModel(const Model &model) {
		model2evaluate = model;
	}

	inline double Error(size_t sample) const {
		return SQUARE(model2evaluate.Distance(points_[sample]));
	}

	inline size_t NumSamples() const { return static_cast<size_t>(points_.GetSize()); }
	inline double logalpha0() const { return logalpha0_; }
	inline double multError() const { return 0.5; }

protected:
	const Point3Arr& points_; // Normalized input data
	double logalpha0_; // Alpha0 is used to make the error adaptive to the image size
	Model model2evaluate; // current model to be evaluated
};

// Robustly estimate the plane that fits best the given points
template <typename Sampler, bool bFixThreshold>
unsigned TEstimatePlane(const Point3Arr& points, Plane& plane, double& maxThreshold, bool arrInliers[], size_t maxIters)
{
	const unsigned nPoints = (unsigned)points.GetSize();
	if (nPoints < PlaneSolverAdaptor::MINIMUM_SAMPLES) {
		ASSERT("too few points" == NULL);
		return 0;
	}

	// normalize points
	Matrix4x4 H;
	Point3Arr normPoints;
	NormalizePoints(points, normPoints, &H);

	// plane robust estimation
	std::vector<size_t> vec_inliers;
	Sampler sampler;
	if (bFixThreshold) {
		PlaneSolverAdaptor kernel(normPoints);
		RANSAC(kernel, sampler, vec_inliers, plane, maxThreshold!=0?maxThreshold*H(0,0):0.35, 0.99, maxIters);
		DEBUG_LEVEL(3, "Robust plane: %u/%u points", vec_inliers.size(), nPoints);
	} else {
		PlaneSolverAdaptor kernel(normPoints, 1, 1, 1);
		const std::pair<double,double> ACRansacOut(ACRANSAC(kernel, sampler, vec_inliers, plane, maxThreshold, 0.99, maxIters));
		const double& thresholdSq = ACRansacOut.first;
		maxThreshold = SQRT(thresholdSq);
		DEBUG_LEVEL(3, "Auto-robust plane: %u/%u points (%g threshold)", vec_inliers.size(), nPoints, maxThreshold/H(0,0));
	}
	const unsigned inliers_count = (unsigned)vec_inliers.size();
	if (inliers_count < PlaneSolverAdaptor::MINIMUM_SAMPLES)
		return 0;

	// fit plane to all the inliers
	Point3Arr normInliers(inliers_count);
	for (uint32_t i=0; i<inliers_count; ++i)
		normInliers[i] = normPoints[vec_inliers[i]];
	FitPlane(normInliers.GetData(), normInliers.GetSize(), plane);
	// if a list of inliers is requested, copy it
	if (arrInliers) {
		memset(arrInliers, 0, sizeof(bool)*nPoints);
		for (uint32_t i=0; i<inliers_count; ++i)
			arrInliers[vec_inliers[i]] = true;
	}

	// un-normalize plane
	plane.m_fD /= H(0,0);
	maxThreshold /= H(0,0);

	return inliers_count;
} // TEstimatePlane

} // namespace MVS

// Robustly estimate the plane that fits best the given points
unsigned MVS::EstimatePlane(const Point3Arr& points, Plane& plane, double& maxThreshold, bool arrInliers[], size_t maxIters)
{
	return TEstimatePlane<UniformSampler,false>(points, plane, maxThreshold, arrInliers, maxIters);
} // EstimatePlane
// Robustly estimate the plane that fits best the given points, making sure the first point is part of the solution (if any)
unsigned MVS::EstimatePlaneLockFirstPoint(const Point3Arr& points, Plane& plane, double& maxThreshold, bool arrInliers[], size_t maxIters)
{
	return TEstimatePlane<UniformSamplerLockFirst,false>(points, plane, maxThreshold, arrInliers, maxIters);
} // EstimatePlaneLockFirstPoint
// Robustly estimate the plane that fits best the given points using a known threshold
unsigned MVS::EstimatePlaneTh(const Point3Arr& points, Plane& plane, double maxThreshold, bool arrInliers[], size_t maxIters)
{
	return TEstimatePlane<UniformSampler,true>(points, plane, maxThreshold, arrInliers, maxIters);
} // EstimatePlaneTh
// Robustly estimate the plane that fits best the given points using a known threshold, making sure the first point is part of the solution (if any)
unsigned MVS::EstimatePlaneThLockFirstPoint(const Point3Arr& points, Plane& plane, double maxThreshold, bool arrInliers[], size_t maxIters)
{
	return TEstimatePlane<UniformSamplerLockFirst,true>(points, plane, maxThreshold, arrInliers, maxIters);
} // EstimatePlaneThLockFirstPoint
/*----------------------------------------------------------------*/


// estimate the colors of the given dense point cloud
void MVS::EstimatePointColors(const ImageArr& images, PointCloud& pointcloud)
{
	TD_TIMER_START();

	pointcloud.colors.Resize(pointcloud.points.GetSize());
	FOREACH(i, pointcloud.colors) {
		PointCloud::Color& color = pointcloud.colors[i];
		const PointCloud::Point& point = pointcloud.points[i];
		const PointCloud::ViewArr& views= pointcloud.pointViews[i];
		// compute vertex color
		REAL bestDistance(FLT_MAX);
		const Image* pImageData(NULL);
		FOREACHPTR(pView, views) {
			const Image& imageData = images[*pView];
			ASSERT(imageData.IsValid());
			if (imageData.image.empty())
				continue;
			// compute the distance from the 3D point to the image
			const REAL distance(imageData.camera.PointDepth(point));
			ASSERT(distance > 0);
			if (bestDistance > distance) {
				bestDistance = distance;
				pImageData = &imageData;
			}
		}
		if (pImageData == NULL) {
			// set a dummy color
			color = Pixel8U::WHITE;
		} else {
			// get image color
			const Point2f proj(pImageData->camera.ProjectPointP(point));
			color = (pImageData->image.isInsideWithBorder<float,1>(proj) ? pImageData->image.sample(proj) : Pixel8U::WHITE);
		}
	}

	DEBUG_ULTIMATE("Estimate dense point cloud colors: %u colors (%s)", pointcloud.colors.GetSize(), TD_TIMER_GET_FMT().c_str());
} // EstimatePointColors
/*----------------------------------------------------------------*/

// estimate the labels dense point cloud
void MVS::EstimatePointLabels(const ImageArr& images, PointCloud& pointcloud)
{
	TD_TIMER_START();

	std::map<String, Image8U3> labels;

	pointcloud.colors.Resize(pointcloud.points.GetSize());
	FOREACH(i, pointcloud.colors) {
		PointCloud::Color& color = pointcloud.colors[i];
		const PointCloud::Point& point = pointcloud.points[i];
		const PointCloud::ViewArr& views= pointcloud.pointViews[i];
		// compute vertex color
		REAL bestDistance(FLT_MAX);
		const Image* pImageData(NULL);
		FOREACHPTR(pView, views) {						
			const Image& imageData = images[*pView];						
			ASSERT(imageData.IsValid());	
			// compute the distance from the 3D point to the image
			const REAL distance(imageData.camera.PointDepth(point));
			ASSERT(distance > 0);
			if (bestDistance > distance) {
				bestDistance = distance;
				pImageData = &imageData;
			}
		}

		// if image not loaded -> insert in map
		if (labels.find(pImageData->coloredMaskName) == labels.end())
		{			
			std::cout << pImageData->coloredMaskName << std::endl;
			labels[pImageData->coloredMaskName].Load(pImageData->coloredMaskName);

			if (labels[pImageData->coloredMaskName].size() != pImageData->GetSize())
			{
				cv::resize(labels[pImageData->coloredMaskName], labels[pImageData->coloredMaskName], pImageData->GetSize(), 0, 0, cv::INTER_NEAREST);
				std::cout << labels[pImageData->coloredMaskName].size() << std::endl;
			}
		}

		if (pImageData == NULL) {
			// set a dummy color
			color = Pixel8U::WHITE;
		} else {
			// get image color
			const Point2f proj(pImageData->camera.ProjectPointP(point));		
			//color = (pImageData->image.isInsideWithBorder<float,1>(proj) ? labels[pImageData->coloredMaskName].sample(proj) : Pixel8U::RED);
			//std::cout << color.r << " " << color.g << " " << color.b <<  std::endl;
			color = (Pixel8U)labels[pImageData->coloredMaskName].sample(proj);
		}
	}

	DEBUG_ULTIMATE("Estimate dense point cloud labels: %u labels (%s)", pointcloud.colors.GetSize(), TD_TIMER_GET_FMT().c_str());
} // EstimatePointLabels
/*----------------------------------------------------------------*/

// estimates the normals through PCA over the K nearest neighbors
void MVS::EstimatePointNormals(const ImageArr& images, PointCloud& pointcloud, int numNeighbors /*K-nearest neighbors*/)
{
	TD_TIMER_START();

	typedef CGAL::Simple_cartesian<double> kernel_t;
	typedef kernel_t::Point_3 point_t;
	typedef kernel_t::Vector_3 vector_t;
	typedef std::pair<point_t,vector_t> PointVectorPair;
	// fetch the point set
	std::vector<PointVectorPair> pointvectors(pointcloud.points.GetSize());
	FOREACH(i, pointcloud.points)
		(Point3d&)(pointvectors[i].first) = pointcloud.points[i];
	// estimates normals direction;
	// Note: pca_estimate_normals() requires an iterator over points
	// as well as property maps to access each point's position and normal.
	#if CGAL_VERSION_NR < 1041301000
	#if CGAL_VERSION_NR < 1040800000
	CGAL::pca_estimate_normals(
	#else
	CGAL::pca_estimate_normals<CGAL::Sequential_tag>(
	#endif
		pointvectors.begin(), pointvectors.end(),
		CGAL::First_of_pair_property_map<PointVectorPair>(),
		CGAL::Second_of_pair_property_map<PointVectorPair>(),
		numNeighbors
	);
	#else
	CGAL::pca_estimate_normals<CGAL::Sequential_tag>(
		pointvectors, numNeighbors,
		CGAL::parameters::point_map(CGAL::First_of_pair_property_map<PointVectorPair>())
		.normal_map(CGAL::Second_of_pair_property_map<PointVectorPair>())
	);
	#endif
	// store the point normals
	pointcloud.normals.Resize(pointcloud.points.GetSize());
	FOREACH(i, pointcloud.normals) {
		PointCloud::Normal& normal = pointcloud.normals[i];
		const PointCloud::Point& point = pointcloud.points[i];
		const PointCloud::ViewArr& views= pointcloud.pointViews[i];
		normal = (const Point3d&)(pointvectors[i].second);
		// correct normal orientation
		ASSERT(!views.IsEmpty());
		const Image& imageData = images[views.First()];
		if (normal.dot(Cast<float>(imageData.camera.C)-point) < 0)
			normal = -normal;
	}

	DEBUG_ULTIMATE("Estimate dense point cloud normals: %u normals (%s)", pointcloud.normals.GetSize(), TD_TIMER_GET_FMT().c_str());
} // EstimatePointNormals
/*----------------------------------------------------------------*/

bool MVS::EstimateNormalMap(const Matrix3x3f& K, const DepthMap& depthMap, NormalMap& normalMap)
{
	normalMap.create(depthMap.size());
	struct Tool {
		static bool IsDepthValid(Depth d, Depth nd) {
			return nd > 0 && IsDepthSimilar(d, nd, Depth(0.03f));
		}
		// computes depth gradient (first derivative) at current pixel
		static bool DepthGradient(const DepthMap& depthMap, const ImageRef& ir, Point3f& ws) {
			float& w  = ws[0];
			float& wx = ws[1];
			float& wy = ws[2];
			w = depthMap(ir);
			if (w <= 0)
				return false;
			// loop over neighborhood and finding least squares plane,
			// the coefficients of which give gradient of depth
			int whxx(0), whxy(0), whyy(0);
			float wgx(0), wgy(0);
			const int Radius(1);
			int n(0);
			for (int y = -Radius; y <= Radius; ++y) {
				for (int x = -Radius; x <= Radius; ++x) {
					if (x == 0 && y == 0)
						continue;
					const ImageRef pt(ir.x+x, ir.y+y);
					if (!depthMap.isInside(pt))
						continue;
					const float wi(depthMap(pt));
					if (!IsDepthValid(w, wi))
						continue;
					whxx += x*x; whxy += x*y; whyy += y*y;
					wgx += (wi - w)*x; wgy += (wi - w)*y;
					++n;
				}
			}
			if (n < 3)
				return false;
			// solve 2x2 system, generated from depth gradient
			const int det(whxx*whyy - whxy*whxy);
			if (det == 0)
				return false;
			const float invDet(1.f/float(det));
			wx = (float( whyy)*wgx - float(whxy)*wgy)*invDet;
			wy = (float(-whxy)*wgx + float(whxx)*wgy)*invDet;
			return true;
		}
		// computes normal to the surface given the depth and its gradient
		static Normal ComputeNormal(const Matrix3x3f& K, int x, int y, Depth d, Depth dx, Depth dy) {
			ASSERT(ISZERO(K(0,1)));
			return normalized(Normal(
				K(0,0)*dx,
				K(1,1)*dy,
				(K(0,2)-float(x))*dx+(K(1,2)-float(y))*dy-d
			));
		}
	};
	for (int r=0; r<normalMap.rows; ++r) {
		for (int c=0; c<normalMap.cols; ++c) {
			#if 0
			const Depth d(depthMap(r,c));
			if (d <= 0) {
				normalMap(r,c) = Normal::ZERO;
				continue;
			}
			Depth dl, du;
			if (depthMap.isInside(ImageRef(c-1,r-1)) && Tool::IsDepthValid(d, dl=depthMap(r,c-1)) &&  Tool::IsDepthValid(d, du=depthMap(r-1,c)))
				normalMap(r,c) = Tool::ComputeNormal(K, c, r, d, du-d, dl-d);
			else
			if (depthMap.isInside(ImageRef(c+1,r-1)) && Tool::IsDepthValid(d, dl=depthMap(r,c+1)) &&  Tool::IsDepthValid(d, du=depthMap(r-1,c)))
				normalMap(r,c) = Tool::ComputeNormal(K, c, r, d, du-d, d-dl);
			else
			if (depthMap.isInside(ImageRef(c+1,r+1)) && Tool::IsDepthValid(d, dl=depthMap(r,c+1)) &&  Tool::IsDepthValid(d, du=depthMap(r+1,c)))
				normalMap(r,c) = Tool::ComputeNormal(K, c, r, d, d-du, d-dl);
			else
			if (depthMap.isInside(ImageRef(c-1,r+1)) && Tool::IsDepthValid(d, dl=depthMap(r,c-1)) &&  Tool::IsDepthValid(d, du=depthMap(r+1,c)))
				normalMap(r,c) = Tool::ComputeNormal(K, c, r, d, d-du, dl-d);
			else
				normalMap(r,c) = Normal(0,0,-1);
			#else
			// calculates depth gradient at x
			Normal& n = normalMap(r,c);
			if (Tool::DepthGradient(depthMap, ImageRef(c,r), n))
				n = Tool::ComputeNormal(K, c, r, n.x, n.y, n.z);
			else
				n = Normal::ZERO;
			#endif
			ASSERT(normalMap(r,c).dot(K.inv()*Point3f(float(c),float(r),1.f)) <= 0);
		}
	}
	return true;
} // EstimateNormalMap
/*----------------------------------------------------------------*/


// save the depth map in our .dmap file format
bool MVS::SaveDepthMap(const String& fileName, const DepthMap& depthMap)
{
	ASSERT(!depthMap.empty());
	return SerializeSave(depthMap, fileName, ARCHIVE_BINARY_ZIP);
} // SaveDepthMap
/*----------------------------------------------------------------*/
// load the depth map from our .dmap file format
bool MVS::LoadDepthMap(const String& fileName, DepthMap& depthMap)
{
	return SerializeLoad(depthMap, fileName, ARCHIVE_BINARY_ZIP);
} // LoadDepthMap
/*----------------------------------------------------------------*/

// save the normal map in our .nmap file format
bool MVS::SaveNormalMap(const String& fileName, const NormalMap& normalMap)
{
	ASSERT(!normalMap.empty());
	return SerializeSave(normalMap, fileName, ARCHIVE_BINARY_ZIP);
} // SaveNormalMap
/*----------------------------------------------------------------*/
// load the normal map from our .nmap file format
bool MVS::LoadNormalMap(const String& fileName, NormalMap& normalMap)
{
	return SerializeLoad(normalMap, fileName, ARCHIVE_BINARY_ZIP);
} // LoadNormalMap
/*----------------------------------------------------------------*/

// save the confidence map in our .cmap file format
bool MVS::SaveConfidenceMap(const String& fileName, const ConfidenceMap& confMap)
{
	ASSERT(!confMap.empty());
	return SerializeSave(confMap, fileName, ARCHIVE_BINARY_ZIP);
} // SaveConfidenceMap
/*----------------------------------------------------------------*/
// load the confidence map from our .cmap file format
bool MVS::LoadConfidenceMap(const String& fileName, ConfidenceMap& confMap)
{
	return SerializeLoad(confMap, fileName, ARCHIVE_BINARY_ZIP);
} // LoadConfidenceMap
/*----------------------------------------------------------------*/



// export depth map as an image (dark - far depth, light - close depth)
Image8U3 MVS::DepthMap2Image(const DepthMap& depthMap, Depth minDepth, Depth maxDepth)
{
	ASSERT(!depthMap.empty());
	// find min and max values
	if (minDepth == FLT_MAX && maxDepth == 0) {
		cList<Depth,Depth,0> depths(0, depthMap.area());
		for (int i=depthMap.area(); --i >= 0; ) {
			const Depth depth = depthMap[i];
			ASSERT(depth == 0 || depth > 0);
			if (depth > 0)
				depths.Insert(depth);
		}
		if (!depths.empty()) {
			const std::pair<Depth,Depth> th(ComputeX84Threshold<Depth,Depth>(depths.data(), depths.size()));
			const std::pair<Depth,Depth> mm(depths.GetMinMax());
			maxDepth = MINF(th.first+th.second, mm.second);
			minDepth = MAXF(th.first-th.second, mm.first);
		}
		DEBUG_ULTIMATE("\tdepth range: [%g, %g]", minDepth, maxDepth);
	}
	const Depth sclDepth(Depth(1)/(maxDepth - minDepth));
	// create color image
	Image8U3 img(depthMap.size());
	for (int i=depthMap.area(); --i >= 0; ) {
		const Depth depth = depthMap[i];
		img[i] = (depth > 0 ? Pixel8U::gray2color(CLAMP((maxDepth-depth)*sclDepth, Depth(0), Depth(1))) : Pixel8U::BLACK);
	}
	return img;
} // DepthMap2Image
bool MVS::ExportDepthMap(const String& fileName, const DepthMap& depthMap, Depth minDepth, Depth maxDepth)
{
	if (depthMap.empty())
		return false;
	return DepthMap2Image(depthMap, minDepth, maxDepth).Save(fileName);
} // ExportDepthMap
/*----------------------------------------------------------------*/




template <typename T>
T Percentile(const std::vector<T>& elems, const double p) {

  const int idx = static_cast<int>(std::round(p / 100 * (elems.size() - 1)));
  const size_t percentile_idx =
      std::max(0, std::min(static_cast<int>(elems.size() - 1), idx));

  std::vector<T> ordered_elems = elems;
  std::nth_element(ordered_elems.begin(),
                   ordered_elems.begin() + percentile_idx, ordered_elems.end());

  return ordered_elems.at(percentile_idx);
}

float MVS::JetColormap::Red(const float gray) { return Base(gray - 0.25f); }

float MVS::JetColormap::Green(const float gray) { return Base(gray); }

float MVS::JetColormap::Blue(const float gray) { return Base(gray + 0.25f); }

float MVS::JetColormap::Base(const float val) {
  if (val <= 0.125f) {
    return 0.0f;
  } else if (val <= 0.375f) {
    return Interpolate(2.0f * val - 1.0f, 0.0f, -0.75f, 1.0f, -0.25f);
  } else if (val <= 0.625f) {
    return 1.0f;
  } else if (val <= 0.87f) {
    return Interpolate(2.0f * val - 1.0f, 1.0f, 0.25f, 0.0f, 0.75f);
  } else {
    return 0.0f;
  }
}

float MVS::JetColormap::Interpolate(const float val, const float y0, const float x0,
                               					const float y1, const float x1) {
  return (val - x0) * (y1 - y0) / (x1 - x0) + y0;
}



bool MVS::ExportDepthDiffmap(const String& fileName, const Image32F& diffMap)
{
	// save image
	Image8U img(diffMap.size());

	const uint32_t width = (uint32_t)diffMap.width();
	const uint32_t height = (uint32_t)diffMap.height();
	const uint32_t max = 2 ;
	const uint32_t min = 0 ;

	for (uint32_t i=0; i<height; ++i) 
	{
		for (uint32_t j=0; j<width; ++j) 
		{
			float diff = diffMap(i,j) ;

			
			if( diff<0 )
			{ diff = 0; }
			if( diff>2 )
			{ diff = 2 ; }

			
			float percent = diff*1.0/(max-min) ;

			
			img(i,j) = int(percent*255);
		}
	}
	return img.Save(fileName);
}


bool MVS::ExportDepthMapByJetColormap(const String& fileName, const DepthMap& depthMap)
{
	// find min and max values
	std::vector<float> valid_depths;
	for (int i=depthMap.area(); --i >= 0; ) {
		const Depth depth = depthMap[i];
		if (depth > 0)
			valid_depths.push_back(depth);
	}
	if (valid_depths.empty()) {
		return false;
	}

	float min_percentile = 2;
	float max_percentile = 98;
	const float robust_depth_min = Percentile(valid_depths, min_percentile);
	const float robust_depth_max = Percentile(valid_depths, max_percentile);
	const float robust_depth_range = robust_depth_max - robust_depth_min;
	// save image
	Image8U3 img(depthMap.size());
	for (int i=depthMap.area(); --i >= 0; ) {
		const Depth depth = depthMap[i];
		if(depth > 0)
		{
			const float robust_depth = std::max(robust_depth_min, std::min(robust_depth_max, depth));
        	const float gray = (robust_depth - robust_depth_min) / robust_depth_range;
			img[i] = Image8U3::Type(
									255 * JetColormap::Red(gray),
									255 * JetColormap::Green(gray),
									255 * JetColormap::Blue(gray));
		}
		else
		{
			img[i] = Image8U3::Type::BLACK;
		}
	}
	return img.Save(fileName);
}


bool MVS::ExportSuperPixelLabelsmap(const String& fileName, const Image16I& labelsMap)
{
	// save image
	Image8U img(labelsMap.size());

	const uint32_t width = (uint32_t)labelsMap.width();
	const uint32_t height = (uint32_t)labelsMap.height();

	for (uint32_t i=0; i<height; ++i) 
	{
		for (uint32_t j=0; j<width; ++j) 
		{
			
			img(i,j) = labelsMap(i,j)%256 ;
		}
	}
	return img.Save(fileName);
}


// export normal map as an image
bool MVS::ExportNormalMap(const String& fileName, const NormalMap& normalMap)
{
	if (normalMap.empty())
		return false;
	Image8U3 img(normalMap.size());
	for (int i=normalMap.area(); --i >= 0; ) {
		img[i] = [](const Normal& n) {
			return ISZERO(n) ?
				Image8U3::Type::BLACK :
				Image8U3::Type(
					CLAMP(ROUND2INT((1.f-n.x)*127.5f), 0, 255),
					CLAMP(ROUND2INT((1.f-n.y)*127.5f), 0, 255),
					CLAMP(ROUND2INT(    -n.z *255.0f), 0, 255)
				);
		} (normalMap[i]);
	}
	return img.Save(fileName);
} // ExportNormalMap
/*----------------------------------------------------------------*/

// export confidence map as an image (dark - low confidence, light - high confidence)
bool MVS::ExportConfidenceMap(const String& fileName, const ConfidenceMap& confMap)
{
	// find min and max values
	FloatArr confs(0, confMap.area());
	for (int i=confMap.area(); --i >= 0; ) {
		const float conf = confMap[i];
		ASSERT(conf == 0 || conf > 0);
		if (conf > 0)
			confs.Insert(conf);
	}
	if (confs.IsEmpty())
		return false;
	const std::pair<float,float> th(ComputeX84Threshold<float,float>(confs.Begin(), confs.GetSize()));
	float minConf = th.first-th.second;
	float maxConf = th.first+th.second;
	if (minConf < 0.1f)
		minConf = 0.1f;
	if (maxConf < 0.1f)
		maxConf = 30.f;
	DEBUG_ULTIMATE("\tconfidence range: [%g, %g]", minConf, maxConf);
	const float deltaConf = maxConf - minConf;
	// save image
	Image8U img(confMap.size());
	for (int i=confMap.area(); --i >= 0; ) {
		const float conf = confMap[i];
		img[i] = (conf > 0 ? (uint8_t)CLAMP((conf-minConf)*255.f/deltaConf, 0.f, 255.f) : 0);
	}
	return img.Save(fileName);
} // ExportConfidenceMap



bool MVS::ExportBinDepth(DepthData& depthData)
{
	std::ofstream ofs;
	ofs.open(ComposeBinDepthPath(depthData.GetView().GetID(), "jpg"), std::ios::out  |  std::ios::binary );
	DepthMap  depthmap(depthData.superpriors_depthMap); 
	
	const uint32_t width = (uint32_t)depthData.depthMap.width();
	const uint32_t height = (uint32_t)depthData.depthMap.height();
	for (uint32_t j=0; j<height; ++j) //纵坐标
	{                
		for (uint32_t i=0; i<width; ++i) //横坐标
		{            
			const  ImageRef x(i,j);
			Depth depth(depthmap(x)); 
			ofs.write((const char *)&depth, sizeof(Depth));
		}
	}
	ofs.close();
}

// export point cloud
bool MVS::ExportPointCloud(const String& fileName, const Image& imageData, const DepthMap& depthMap, const NormalMap& normalMap)
{
	ASSERT(!depthMap.empty());
	const Camera& P0 = imageData.camera;
	if (normalMap.empty()) {
		// vertex definition
		struct Vertex {
			float x,y,z;
			uint8_t r,g,b;
		};
		// list of property information for a vertex
		static PLY::PlyProperty vert_props[] = {
			{"x", PLY::Float32, PLY::Float32, offsetof(Vertex,x), 0, 0, 0, 0},
			{"y", PLY::Float32, PLY::Float32, offsetof(Vertex,y), 0, 0, 0, 0},
			{"z", PLY::Float32, PLY::Float32, offsetof(Vertex,z), 0, 0, 0, 0},
			{"red", PLY::Uint8, PLY::Uint8, offsetof(Vertex,r), 0, 0, 0, 0},
			{"green", PLY::Uint8, PLY::Uint8, offsetof(Vertex,g), 0, 0, 0, 0},
			{"blue", PLY::Uint8, PLY::Uint8, offsetof(Vertex,b), 0, 0, 0, 0},
		};
		// list of the kinds of elements in the PLY
		static const char* elem_names[] = {
			"vertex"
		};

		// create PLY object
		ASSERT(!fileName.IsEmpty());
		Util::ensureFolder(fileName);
		const size_t bufferSize = depthMap.area()*(8*3/*pos*/+3*3/*color*/+7/*space*/+2/*eol*/) + 2048/*extra size*/;
		PLY ply;
		if (!ply.write(fileName, 1, elem_names, PLY::BINARY_LE, bufferSize))
			return false;

		// describe what properties go into the vertex elements
		ply.describe_property("vertex", 6, vert_props);

		// export the array of 3D points
		Vertex vertex;
		for (int j=0; j<depthMap.rows; ++j) {
			for (int i=0; i<depthMap.cols; ++i) {
				const Depth& depth = depthMap(j,i);
				ASSERT(depth >= 0);
				if (depth <= 0)
					continue;
				const Point3f X(P0.TransformPointI2W(Point3(i,j,depth)));
				vertex.x = X.x; vertex.y = X.y; vertex.z = X.z;
				const Pixel8U c(imageData.image.empty() ? Pixel8U::WHITE : imageData.image(j,i));
				vertex.r = c.r; vertex.g = c.g; vertex.b = c.b;
				ply.put_element(&vertex);
			}
		}
		if (ply.get_current_element_count() == 0)
			return false;

		// write to file
		if (!ply.header_complete())
			return false;
	} else {
		// vertex definition
		struct Vertex {
			float x,y,z;
			float nx,ny,nz;
			uint8_t r,g,b;
		};
		// list of property information for a vertex
		static PLY::PlyProperty vert_props[] = {
			{"x", PLY::Float32, PLY::Float32, offsetof(Vertex,x), 0, 0, 0, 0},
			{"y", PLY::Float32, PLY::Float32, offsetof(Vertex,y), 0, 0, 0, 0},
			{"z", PLY::Float32, PLY::Float32, offsetof(Vertex,z), 0, 0, 0, 0},
			{"nx", PLY::Float32, PLY::Float32, offsetof(Vertex,nx), 0, 0, 0, 0},
			{"ny", PLY::Float32, PLY::Float32, offsetof(Vertex,ny), 0, 0, 0, 0},
			{"nz", PLY::Float32, PLY::Float32, offsetof(Vertex,nz), 0, 0, 0, 0},
			{"red", PLY::Uint8, PLY::Uint8, offsetof(Vertex,r), 0, 0, 0, 0},
			{"green", PLY::Uint8, PLY::Uint8, offsetof(Vertex,g), 0, 0, 0, 0},
			{"blue", PLY::Uint8, PLY::Uint8, offsetof(Vertex,b), 0, 0, 0, 0},
		};
		// list of the kinds of elements in the PLY
		static const char* elem_names[] = {
			"vertex"
		};

		// create PLY object
		ASSERT(!fileName.IsEmpty());
		Util::ensureFolder(fileName);
		const size_t bufferSize = depthMap.area()*(8*3/*pos*/+8*3/*normal*/+3*3/*color*/+8/*space*/+2/*eol*/) + 2048/*extra size*/;
		PLY ply;
		if (!ply.write(fileName, 1, elem_names, PLY::BINARY_LE, bufferSize))
			return false;

		// describe what properties go into the vertex elements
		ply.describe_property("vertex", 9, vert_props);

		// export the array of 3D points
		Vertex vertex;
		for (int j=0; j<depthMap.rows; ++j) {
			for (int i=0; i<depthMap.cols; ++i) {
				const Depth& depth = depthMap(j,i);
				ASSERT(depth >= 0);
				if (depth <= 0)
					continue;
				const Point3f X(P0.TransformPointI2W(Point3(i,j,depth)));
				vertex.x = X.x; vertex.y = X.y; vertex.z = X.z;
				const Point3f N(P0.R.t() * Cast<REAL>(normalMap(j,i)));
				vertex.nx = N.x; vertex.ny = N.y; vertex.nz = N.z;
				const Pixel8U c(imageData.image.empty() ? Pixel8U::WHITE : imageData.image(j, i));
				vertex.r = c.r; vertex.g = c.g; vertex.b = c.b;
				ply.put_element(&vertex);
			}
		}
		if (ply.get_current_element_count() == 0)
			return false;

		// write to file
		if (!ply.header_complete())
			return false;
	}
	return true;
} // ExportPointCloud
/*----------------------------------------------------------------*/


bool MVS::ExportDepthDataRaw(const String& fileName, const String& imageFileName,
	const IIndexArr& IDs, const cv::Size& imageSize,
	const KMatrix& K, const RMatrix& R, const CMatrix& C,
	Depth dMin, Depth dMax,
	const DepthMap& depthMap, const NormalMap& normalMap, const ConfidenceMap& confMap)
{
	ASSERT(!depthMap.empty());
	ASSERT(confMap.empty() || depthMap.size() == confMap.size());
	ASSERT(depthMap.width() <= imageSize.width && depthMap.height() <= imageSize.height);

	FILE *f = fopen(fileName, "wb");
	if (f == NULL) {
		DEBUG("error: opening file '%s' for writing depth-data", fileName.c_str());
		return false;
	}

	// write header
	HeaderDepthDataRaw header;
	header.name = HeaderDepthDataRaw::HeaderDepthDataRawName();
	header.type = HeaderDepthDataRaw::HAS_DEPTH;
	header.imageWidth = (uint32_t)imageSize.width;
	header.imageHeight = (uint32_t)imageSize.height;
	header.depthWidth = (uint32_t)depthMap.cols;
	header.depthHeight = (uint32_t)depthMap.rows;
	header.dMin = dMin;
	header.dMax = dMax;
	if (!normalMap.empty())
		header.type |= HeaderDepthDataRaw::HAS_NORMAL;
	if (!confMap.empty())
		header.type |= HeaderDepthDataRaw::HAS_CONF;
	fwrite(&header, sizeof(HeaderDepthDataRaw), 1, f);

	// write image file name
	STATIC_ASSERT(sizeof(String::value_type) == sizeof(char));
	const String FileName(MAKE_PATH_REL(Util::getFullPath(Util::getFilePath(fileName)), Util::getFullPath(imageFileName)));
	const uint16_t nFileNameSize((uint16_t)FileName.length());
	fwrite(&nFileNameSize, sizeof(uint16_t), 1, f);
	fwrite(FileName.c_str(), sizeof(char), nFileNameSize, f);

	// write neighbor IDs
	STATIC_ASSERT(sizeof(uint32_t) == sizeof(IIndex));
	const uint32_t nIDs(IDs.size());
	fwrite(&nIDs, sizeof(IIndex), 1, f);
	fwrite(IDs.data(), sizeof(IIndex), nIDs, f);

	// write pose
	STATIC_ASSERT(sizeof(double) == sizeof(REAL));
	fwrite(K.val, sizeof(REAL), 9, f);
	fwrite(R.val, sizeof(REAL), 9, f);
	fwrite(C.ptr(), sizeof(REAL), 3, f);

	// write depth-map
	fwrite(depthMap.getData(), sizeof(float), depthMap.area(), f);

	// write normal-map
	if ((header.type & HeaderDepthDataRaw::HAS_NORMAL) != 0)
		fwrite(normalMap.getData(), sizeof(float)*3, normalMap.area(), f);

	// write confidence-map
	if ((header.type & HeaderDepthDataRaw::HAS_CONF) != 0)
		fwrite(confMap.getData(), sizeof(float), confMap.area(), f);

	const bool bRet(ferror(f) == 0);
	fclose(f);
	return bRet;
} // ExportDepthDataRaw

bool MVS::ImportDepthDataRaw(const String& fileName, String& imageFileName,
	IIndexArr& IDs, cv::Size& imageSize,
	KMatrix& K, RMatrix& R, CMatrix& C,
	Depth& dMin, Depth& dMax,
	DepthMap& depthMap, NormalMap& normalMap, ConfidenceMap& confMap, unsigned flags)
{
	FILE *f = fopen(fileName, "rb");
	if (f == NULL) {
		DEBUG("error: opening file '%s' for reading depth-data", fileName.c_str());
		return false;
	}

	// read header
	HeaderDepthDataRaw header;
	if (fread(&header, sizeof(HeaderDepthDataRaw), 1, f) != 1 ||
		header.name != HeaderDepthDataRaw::HeaderDepthDataRawName() ||
		(header.type & HeaderDepthDataRaw::HAS_DEPTH) == 0 ||
		header.depthWidth <= 0 || header.depthHeight <= 0 ||
		header.imageWidth < header.depthWidth || header.imageHeight < header.depthHeight)
	{
		DEBUG("error: invalid depth-data file '%s'", fileName.c_str());
		return false;
	}

	// read image file name
	STATIC_ASSERT(sizeof(String::value_type) == sizeof(char));
	uint16_t nFileNameSize;
	fread(&nFileNameSize, sizeof(uint16_t), 1, f);
	imageFileName.resize(nFileNameSize);
	fread(&imageFileName[0u], sizeof(char), nFileNameSize, f);

	// read neighbor IDs
	STATIC_ASSERT(sizeof(uint32_t) == sizeof(IIndex));
	uint32_t nIDs;
	fread(&nIDs, sizeof(IIndex), 1, f);
	IDs.resize(nIDs);
	fread(IDs.data(), sizeof(IIndex), nIDs, f);

	// read pose
	STATIC_ASSERT(sizeof(double) == sizeof(REAL));
	fread(K.val, sizeof(REAL), 9, f);
	fread(R.val, sizeof(REAL), 9, f);
	fread(C.ptr(), sizeof(REAL), 3, f);

	// read depth-map
	dMin = header.dMin;
	dMax = header.dMax;
	imageSize.width = header.imageWidth;
	imageSize.height = header.imageHeight;
	if ((flags & HeaderDepthDataRaw::HAS_DEPTH) != 0) {
		depthMap.create(header.depthHeight, header.depthWidth);
		fread(depthMap.getData(), sizeof(float), depthMap.area(), f);
	} else {
		fseek(f, sizeof(float)*header.depthWidth*header.depthHeight, SEEK_CUR);
	}

	// read normal-map
	if ((header.type & HeaderDepthDataRaw::HAS_NORMAL) != 0) {
		if ((flags & HeaderDepthDataRaw::HAS_NORMAL) != 0) {
			normalMap.create(header.depthHeight, header.depthWidth);
			fread(normalMap.getData(), sizeof(float)*3, normalMap.area(), f);
		} else {
			fseek(f, sizeof(float)*3*header.depthWidth*header.depthHeight, SEEK_CUR);
		}
	}

	// read confidence-map
	if ((header.type & HeaderDepthDataRaw::HAS_CONF) != 0) {
		if ((flags & HeaderDepthDataRaw::HAS_CONF) != 0) {
			confMap.create(header.depthHeight, header.depthWidth);
			fread(confMap.getData(), sizeof(float), confMap.area(), f);
		}
	}

	const bool bRet(ferror(f) == 0);
	fclose(f);
	return bRet;
} // ImportDepthDataRaw

/*----------------------------------------------------------------*/


// compare the estimated and ground-truth depth-maps
void MVS::CompareDepthMaps(const DepthMap& depthMap, const DepthMap& depthMapGT, uint32_t idxImage, float threshold)
{
	TD_TIMER_START();
	const uint32_t width = (uint32_t)depthMap.width();
	const uint32_t height = (uint32_t)depthMap.height();
	// compute depth errors for each pixel
	cv::resize(depthMapGT, depthMapGT, depthMap.size());
	unsigned nErrorPixels(0);
	unsigned nExtraPixels(0);
	unsigned nMissingPixels(0);
	FloatArr depths(0, depthMap.area());
	FloatArr depthsGT(0, depthMap.area());
	FloatArr errors(0, depthMap.area());
	for (uint32_t i=0; i<height; ++i) {
		for (uint32_t j=0; j<width; ++j) {
			const Depth& depth = depthMap(i,j);
			const Depth& depthGT = depthMapGT(i,j);
			if (depth != 0 && depthGT == 0) {
				++nExtraPixels;
				continue;
			}
			if (depth == 0 && depthGT != 0) {
				++nMissingPixels;
				continue;
			}
			depths.Insert(depth);
			depthsGT.Insert(depthGT);
			const float error(depthGT==0 ? 0 : ABS(depth-depthGT)/depthGT);
			errors.Insert(error);
		}
	}
	const float fPSNR((float)ComputePSNR(DMatrix32F((int)depths.GetSize(),1,depths.GetData()), DMatrix32F((int)depthsGT.GetSize(),1,depthsGT.GetData())));
	const MeanStd<float,double> ms(errors.Begin(), errors.GetSize());
	const float mean((float)ms.GetMean());
	const float stddev((float)ms.GetStdDev());
	const std::pair<float,float> th(ComputeX84Threshold<float,float>(errors.Begin(), errors.GetSize()));
	#if TD_VERBOSE != TD_VERBOSE_OFF
	IDX idxPixel = 0;
	Image8U3 errorsVisual(depthMap.size());
	for (uint32_t i=0; i<height; ++i) {
		for (uint32_t j=0; j<width; ++j) {
			Pixel8U& pix = errorsVisual(i,j);
			const Depth& depth = depthMap(i,j);
			const Depth& depthGT = depthMapGT(i,j);
			if (depth != 0 && depthGT == 0) {
				pix = Pixel8U::GREEN;
				continue;
			}
			if (depth == 0 && depthGT != 0) {
				pix = Pixel8U::BLUE;
				continue;
			}
			const float error = errors[idxPixel++];
			if (depth == 0 && depthGT == 0) {
				pix = Pixel8U::BLACK;
				continue;
			}
			if (error > threshold) {
				pix = Pixel8U::RED;
				++nErrorPixels;
				continue;
			}
			const uint8_t gray((uint8_t)CLAMP((1.f-SAFEDIVIDE(ABS(error), threshold))*255.f, 0.f, 255.f));
			pix = Pixel8U(gray, gray, gray);
		}
	}
	errorsVisual.Save(ComposeDepthFilePath(idxImage, "errors.png"));
	#endif
	VERBOSE("Depth-maps compared for image % 3u: %.4f PSNR; %g median %g mean %g stddev error; %u (%.2f%%%%) error %u (%.2f%%%%) missing %u (%.2f%%%%) extra pixels (%s)",
		idxImage,
		fPSNR,
		th.first, mean, stddev,
		nErrorPixels, (float)nErrorPixels*100.f/depthMap.area(),
		nMissingPixels, (float)nMissingPixels*100.f/depthMap.area(),
		nExtraPixels, (float)nExtraPixels*100.f/depthMap.area(),
		TD_TIMER_GET_FMT().c_str()
	);
}

// compare the estimated and ground-truth normal-maps
void MVS::CompareNormalMaps(const NormalMap& normalMap, const NormalMap& normalMapGT, uint32_t idxImage)
{
	TD_TIMER_START();
	// load normal data
	const uint32_t width = (uint32_t)normalMap.width();
	const uint32_t height = (uint32_t)normalMap.height();
	// compute normal errors for each pixel
	cv::resize(normalMapGT, normalMapGT, normalMap.size());
	FloatArr errors(0, normalMap.area());
	for (uint32_t i=0; i<height; ++i) {
		for (uint32_t j=0; j<width; ++j) {
			const Normal& normal = normalMap(i,j);
			const Normal& normalGT = normalMapGT(i,j);
			if (normal != Normal::ZERO && normalGT == Normal::ZERO)
				continue;
			if (normal == Normal::ZERO && normalGT != Normal::ZERO)
				continue;
			if (normal == Normal::ZERO && normalGT == Normal::ZERO) {
				errors.Insert(0.f);
				continue;
			}
			ASSERT(ISEQUAL(norm(normal),1.f) && ISEQUAL(norm(normalGT),1.f));
			const float error(FR2D(ACOS(CLAMP(normal.dot(normalGT), -1.f, 1.f))));
			errors.Insert(error);
		}
	}
	const MeanStd<float,double> ms(errors.Begin(), errors.GetSize());
	const float mean((float)ms.GetMean());
	const float stddev((float)ms.GetStdDev());
	const std::pair<float,float> th(ComputeX84Threshold<float,float>(errors.Begin(), errors.GetSize()));
	VERBOSE("Normal-maps compared for image % 3u: %.2f median %.2f mean %.2f stddev error (%s)",
		idxImage,
		th.first, mean, stddev,
		TD_TIMER_GET_FMT().c_str()
	);
}
/*----------------------------------------------------------------*/
