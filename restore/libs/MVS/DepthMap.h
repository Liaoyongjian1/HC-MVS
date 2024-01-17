/*
* DepthMap.h
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

#ifndef _MVS_DEPTHMAP_H_
#define _MVS_DEPTHMAP_H_


// I N C L U D E S /////////////////////////////////////////////////

#include "Image.h"
#include "PointCloud.h"

// D E F I N E S ///////////////////////////////////////////////////

// NCC type used for patch-similarity computation during depth-map estimation
#define DENSE_NCC_DEFAULT 0
#define DENSE_NCC_FAST 1
#define DENSE_NCC_WEIGHTED 2
#define DENSE_NCC DENSE_NCC_WEIGHTED

// NCC score aggregation type used during depth-map estimation
#define DENSE_AGGNCC_NTH 0
#define DENSE_AGGNCC_MEAN 1
#define DENSE_AGGNCC_MIN 2
#define DENSE_AGGNCC_MINMEAN 3
#define DENSE_AGGNCC DENSE_AGGNCC_MINMEAN

// type of smoothness used during depth-map estimation
#define DENSE_SMOOTHNESS_NA 0
#define DENSE_SMOOTHNESS_FAST 1
#define DENSE_SMOOTHNESS_PLANE 2
#define DENSE_SMOOTHNESS DENSE_SMOOTHNESS_PLANE

// type of refinement used during depth-map estimation
#define DENSE_REFINE_ITER 0
#define DENSE_REFINE_EXACT 1
#define DENSE_REFINE DENSE_REFINE_ITER

// exp function type used during depth estimation
#define DENSE_EXP_DEFUALT EXP
#define DENSE_EXP_FAST FEXP<true> // ~10% faster, but slightly less precise
#define DENSE_EXP DENSE_EXP_DEFUALT

#define ComposeDepthFilePathBase(b, i, e) MAKE_PATH(String::FormatString((b + "%04u." e).c_str(), i))
#define ComposeDepthFilePath(i, e) MAKE_PATH(String::FormatString("depth%04u." e, i))		
#define ComposeNormalFilePath(i, e) MAKE_PATH(String::FormatString("normal%04u." e, i))	

#define ComposeResizeDepthFilePath(i, e) MAKE_PATH(String::FormatString("depthmap/depth%04u." e, i))			
#define ComposeResizeNormalFilePath(i, e) MAKE_PATH(String::FormatString("normalmap/normal%04u." e, i))		

#define ComposeReadDepthFilePath(i, e) MAKE_PATH(String::FormatString("depthmap/depth%04u." e, i))		
#define ComposeReadNormalFilePath(i, e) MAKE_PATH(String::FormatString("normalmap/normal%04u." e, i))	

#define ComposeDepthFilePropagatedPath(i, e) MAKE_PATH(String::FormatString("propagated%04u." e, i))
#define ComposeInterFramePath(i, e) MAKE_PATH(String::FormatString("InterFrame%04u." e, i))          

#define ComposeDepthFilepriorPath(i, e) MAKE_PATH(String::FormatString("prior%04u." e, i))
#define ComposeDepthFilepriorprogatedPath(i, e) MAKE_PATH(String::FormatString("priorpropagated%04u." e, i))
#define ComposeDepthFileSegmentPath(i, e) MAKE_PATH(String::FormatString("segment%04u." e, i))
#define ComposeDepthDiffPath(i, e) MAKE_PATH(String::FormatString("depthdiff%04u." e, i)) 	
#define ComposeNormalDiffPath(i, e) MAKE_PATH(String::FormatString("normaldiff%04u." e, i)) 	
#define ComposeScoreGraPath(i, e) MAKE_PATH(String::FormatString("Scoregra%04u." e, i)) 		
#define ComposeGraPath(i, e) MAKE_PATH(String::FormatString("Gra%04u." e, i)) 		


#define ComposeMeanshiftDepthPriorsPath(i, e) MAKE_PATH(String::FormatString("meanshiftdepthpriors/meanshiftdepthpriors%04u." e, i)) 	
#define ComposeMeanshiftNormalPriorsPath(i, e) MAKE_PATH(String::FormatString("meanshiftdepthpriors/meanshiftnormalpriors%04u." e, i)) 


#define ComposeSuperPixelLabelsPath(i, e) MAKE_PATH(String::FormatString("labels%04u." e, i))      			
#define ComposeSuperDepthPriorsPath(i, e) MAKE_PATH(String::FormatString("superdepthpriors/superdepthpriors%04u." e, i)) 	
#define ComposeSuperNormalPriorsPath(i, e) MAKE_PATH(String::FormatString("supernormalpriors/supernormalpriors%04u." e, i)) 	
#define ComposeBinDepthPath(i, e) MAKE_PATH(String::FormatString("resize2_superpriors%u." e, i))   			
#define ComposeDepthInFusePath(i, e) MAKE_PATH(String::FormatString("depthfuse%04u." e, i))
#define ComposeNormalInFusePath(i, e) MAKE_PATH(String::FormatString("normalfuse%04u." e, i))


// S T R U C T S ///////////////////////////////////////////////////

namespace MVS {

DECOPT_SPACE(OPTDENSE)

namespace OPTDENSE {
enum DepthFlags {
	REMOVE_SPECKLES	= (1 << 0),
	FILL_GAPS		= (1 << 1),
	ADJUST_FILTER	= (1 << 2),
	OPTIMIZE		= (REMOVE_SPECKLES|FILL_GAPS)
};
extern unsigned nResolutionLevel;
extern unsigned nMaxResolution;
extern unsigned nMinResolution;
extern unsigned nMinViews;
extern unsigned nMaxViews;
extern unsigned nMinViewsFuse;
extern unsigned nMinViewsFilter;
extern unsigned nMinViewsFilterAdjust;
extern unsigned nMinViewsTrustPoint;
extern unsigned nNumViews;
extern bool bFilterAdjust;
extern bool bAddCorners;
extern float fViewMinScore;
extern float fViewMinScoreRatio;
extern float fMinArea;
extern float fMinAngle;
extern float fOptimAngle;
extern float fMaxAngle;
extern float fDescriptorMinMagnitudeThreshold;
extern float fDepthDiffThreshold;
extern float fNormalDiffThreshold;
extern float fPairwiseMul;
extern float fOptimizerEps;
extern int nOptimizerMaxIters;
extern unsigned nSpeckleSize;
extern unsigned nIpolGapSize;
extern String nIgnoreMaskLabel;
extern bool nUseSemantic;
extern unsigned nOptimize;
extern unsigned nEstimateColors;
extern unsigned nEstimateNormals;
extern unsigned ProjectLabels;
extern float fNCCThresholdKeep;
extern unsigned nEstimationIters;
extern unsigned nRandomIters;
extern unsigned nRandomMaxScale;
extern float fRandomDepthRatio;
extern float fRandomAngle1Range;
extern float fRandomAngle2Range;
extern float fRandomSmoothDepth;
extern float fRandomSmoothNormal;
extern float fRandomSmoothBonus;
extern float fSemanticConsistencyMul;
extern float fsigmaTexture;
extern float fsigmaPrior;
extern float fransacEpsilonMul;
extern float fransacClusterMul;
extern float fransacMinPointsDiv;

extern float ransacprobability;

extern float  depthweight ; 	
extern float  normalweight ; 	


extern unsigned nEstimationIters_external;
extern unsigned photo2geo;   
extern float maxgeo_proportion;   
extern float txthreshold;   
extern float txthreshold2;   
extern float para_part;   
extern float para_part2;   
extern float para_tapa;   
extern float para_tapa2;   
extern float para_prior;   
extern float para_prior2;   
extern float photometric_flow; 
extern unsigned initTriangulate;

extern unsigned usepartconsistency;   
extern unsigned usegeoconsistency;  
extern unsigned viewspread;   
extern unsigned opticalflow;  

extern unsigned adapthalfwin;  
extern unsigned propagatehalfwin;  
extern unsigned propagatestep;	


} // namespace OPTDENSE
/*----------------------------------------------------------------*/


template <int nTexels>
struct WeightedPatchFix {
	struct Pixel {
		float weight;
		float tempWeight;
	};
	Pixel weights[nTexels];
	float sumWeights;
	float normSq0;
	WeightedPatchFix() : normSq0(0) {}
};

struct MVS_API DepthData {
	struct ViewData {
		float scale; // image scale relative to the reference image
		Camera camera; // camera matrix corresponding to this image
		Image32F image; // image float intensities
		Image* pImageData; // image data
		DepthMap depthMapPrior; // diventa DepthMap
		NormalMap normalMapPrior;
		flow_ImageRef x0;


		inline IIndex GetID() const {
			return pImageData->ID;
		}
		inline IIndex GetLocalID(const ImageArr& images) const {
			return (IIndex)(pImageData - images.begin());
		}

		template <typename IMAGE>
		static bool ScaleImage(const IMAGE& image, IMAGE& imageScaled, float scale) {
			if (ABS(scale-1.f) < 0.15f)
				return false;
			cv::resize(image, imageScaled, cv::Size(), scale, scale, scale>1?cv::INTER_CUBIC:cv::INTER_AREA);
			return true;
		}
	};
	typedef CLISTDEF2IDX(ViewData,IIndex) ViewDataArr;

	ViewDataArr flow_images; 
	ViewDataArr images; // array of images used to compute this depth-map (reference image is the first)
	ViewScoreArr neighbors; // array of all images seeing this depth-map (ordered by decreasing importance)
	IndexArr points; // indices of the sparse 3D points seen by the this image
	BitMatrix mask; // mark pixels to be ignored
	DepthMap depthMap; // depth-map
	NormalMap normalMap; // normal-map in camera space
	ConfidenceMap confMap; // confidence-map
	Image8U labels;		 
	Image8U3 labelsRGB;	 
	float dMin, dMax; // global depth range for this image
	float delta_c2pmax;   
	unsigned references; // how many times this depth-map is referenced (on 0 can be safely unloaded)
	CriticalSection cs; // used to count references
	
	struct Gra
	{
		int gramax = 100;
		int gramin = 100;
		Point2i point = Point2i(0,0); 
	};
	Image8U graMap;   
	Gra gra ;		  
	Image32F diffMap; 
	Image32F ndiffMap; 
	Image32F scoreGraMap ; 

	struct represent
	{
		Depth depth ;
		Normal normal ;
		float confidence ;
	    Point2i x_rep;
	};
	struct represent_rgb         
	{
        uint8_t r;                           
		uint8_t g;
		uint8_t b;
	};
	int16_t	labelsNum ;				
	Image16I superlabelMap ;		
	std::vector<std::vector<Point2i>> segmentLab ; 
	std::vector<represent> superRepre ; 		    
	std::vector<represent_rgb> superRepre_rgb ;		

	DepthMap superpriors_depthMap; 					
	NormalMap superpriors_normalMap;				

	DepthMap resize_depthMap; 					
	NormalMap resize_normalMap;				

	DepthMap nresize_depthMap; 					
	NormalMap nresize_normalMap;				

	DepthMap meanshiftpriors_depthMap; 				
	NormalMap meanshiftpriors_normalMap;				

	DepthMap depthMap_fuse; 					
	NormalMap normalMap_fuse;				
	inline DepthData() : references(0) {}

	inline void ReleaseImages() {
		FOREACHPTR(ptrImage, images)
			ptrImage->image.release();
	}
	inline void Release() { 
		depthMap.release();
		normalMap.release();
		confMap.release();
	}

	inline bool IsValid() const {
		return !images.IsEmpty();
	}
	inline bool IsEmpty() const {
		return depthMap.empty();
	}

	const ViewData& GetView() const { return images.front(); }
	const Camera& GetCamera() const { return GetView().camera; }

	void GetNormal(const ImageRef& ir, Point3f& N, const TImage<Point3f>* pPointMap=NULL) const;
	void GetNormal(const Point2f& x, Point3f& N, const TImage<Point3f>* pPointMap=NULL) const;

	void ApplyIgnoreMask(const BitMatrix&);

	bool Save(const String& fileName) const;
	bool Load(const String& fileName);

	unsigned GetRef();
	unsigned IncRef(const String& fileName);
	unsigned DecRef();

	#ifdef _USE_BOOST
	// implement BOOST serialization
	template<class Archive>
	void serialize(Archive& ar, const unsigned int /*version*/) {
		ASSERT(IsValid());
		ar & depthMap;
		ar & normalMap;
		ar & confMap;
		ar & dMin;
		ar & dMax;
	}
	#endif
};
typedef MVS_API CLISTDEFIDX(DepthData,IIndex) DepthDataArr;
/*----------------------------------------------------------------*/


struct MVS_API DepthEstimator {
	int16_t adapthalfwin ;	
	enum { nSizeHalfWindow = 7 }; // default was 5
	enum { nSizeWindow = nSizeHalfWindow*2+1 };
	enum { nSizeStep = 2 };
	enum { TexelChannels = 1 };
	enum { nTexels = SQUARE((nSizeHalfWindow*2+nSizeStep)/nSizeStep)*TexelChannels };

	enum ENDIRECTION {
		LT2RB = 0,
		RB2LT,
		DIRS
	};

	typedef TPoint2<uint16_t> MapRef;
	typedef CLISTDEF0(MapRef) MapRefArr;

	typedef Eigen::Matrix<float,nTexels,1> TexelVec;
	struct NeighborData {
		ImageRef x;
		Depth depth;
		Normal normal;
	};
	#if DENSE_SMOOTHNESS != DENSE_SMOOTHNESS_NA
	struct NeighborEstimate {
		Depth depth;
		Normal normal;
		#if DENSE_SMOOTHNESS == DENSE_SMOOTHNESS_PLANE
		Planef::POINT X;
		#endif
	};
	#endif
	#if DENSE_REFINE == DENSE_REFINE_EXACT
	struct PixelEstimate {
		Depth depth;
		Normal normal;
	};
	#endif

	#if DENSE_NCC == DENSE_NCC_WEIGHTED
	typedef WeightedPatchFix<nTexels> Weight;
	typedef CLISTDEFIDX(Weight,int) WeightMap;
	#endif

	// struct ViewData {
	// 	const DepthData::ViewData& view;
	// 	const Matrix3x3 Hl;   //
	// 	const Vec3 Hm;	      // constants during per-pixel loops
	// 	const Matrix3x3 Hr;   //
	// 	inline ViewData() : view(*((const DepthData::ViewData*)this)) {}
	// 	inline ViewData(const DepthData::ViewData& image0, const DepthData::ViewData& image1)
	// 		: view(image1),
	// 		Hl(image1.camera.K * image1.camera.R * image0.camera.R.t()),
	// 		Hm(image1.camera.K * image1.camera.R * (image0.camera.C - image1.camera.C)),
	// 		Hr(image0.camera.K.inv()) {}

	// };

	// Hl=Kj*Rj*inv(Ri), Hm=Kj*Rj*(Ci-Cj),Hr=inv(Ki)
	struct ViewData {
		const DepthData::ViewData& view;
		const Matrix3x3 Hl;    //
		const Vec3 Hm;	         // constants during per-pixel loops
		const Matrix3x3 Hr;   //
		const Matrix3x3 Hl1;    //
		const Vec3 Hm1;	         // constants during per-pixel loops
		const Matrix3x3 Hr1; 
		const Matrix3x3 R; 
		const Vec3 	T;  
		const Matrix3x3 T_ ;
		const Matrix3x3 E; 
		inline ViewData() : view(*((const DepthData::ViewData*)this)) {}
		inline ViewData(const DepthData::ViewData& image0, const DepthData::ViewData& image1)
			: view(image1),
			Hl(image1.camera.K * image1.camera.R * image0.camera.R.t()),
			Hm(image1.camera.K * image1.camera.R * (image0.camera.C - image1.camera.C)),
			Hr(image0.camera.K.inv()),
			Hl1(image0.camera.K * image0.camera.R * image1.camera.R.t()),
			Hm1(image0.camera.K * image0.camera.R * (image1.camera.C - image0.camera.C)),
			Hr1(image1.camera.K.inv()), 
			T(image0.camera.R * (image1.camera.C - image0.camera.C)),
			R(image0.camera.R * image1.camera.R.inv()),
			T_(0,-T[2],T[1],T[2],0,-T[0],-T[1],T[0],0),
			// {T_[0][0]=T_[1][1]=T_[2][2]=0,
			// T_[0][1]=-T[2],T_[1][0]=T[2],
			// T_[2][0]=-T[1],T_[0][2]=T[1],
			// T_[2][1]=T[0],T_[1][2]=-T[0]},
			E(T_*R){}
			
	};
	
	SEACAVE::Random rnd;

	volatile Thread::safe_t& idxPixel; // current image index to be processed
	#if DENSE_SMOOTHNESS == DENSE_SMOOTHNESS_NA
	CLISTDEF0IDX(NeighborData,IIndex) neighbors; // neighbor pixels coordinates to be processed
	#else
	CLISTDEF0IDX(ImageRef,IIndex) neighbors; // neighbor pixels coordinates to be processed
	#endif

	CLISTDEF0IDX(ImageRef,IIndex) candidate; 
	
	#if DENSE_SMOOTHNESS != DENSE_SMOOTHNESS_NA
	CLISTDEF0IDX(NeighborEstimate,IIndex) neighborsClose; // close neighbor pixel depths to be used for smoothing
	#endif
	Vec3 X0;	      //
	ImageRef x0;	  // constants during one pixel loop
	float normSq0;	  //
	#if DENSE_NCC != DENSE_NCC_WEIGHTED
	TexelVec texels0; //
	#endif
	#if DENSE_NCC == DENSE_NCC_DEFAULT
	TexelVec texels1;
	#endif
	#if DENSE_AGGNCC == DENSE_AGGNCC_NTH || DENSE_AGGNCC == DENSE_AGGNCC_MINMEAN
	FloatArr scores;
	#else
	Eigen::VectorXf scores;
	#endif
	#if DENSE_SMOOTHNESS == DENSE_SMOOTHNESS_PLANE
	Planef plane; // plane defined by current depth and normal estimate
	#endif
	DepthMap& depthMap0;
	NormalMap& normalMap0;
	ConfidenceMap& confMap0;
	DepthDataArr& arrDepthData0; 
	WeightMap& weightMap0;
	#endif

	const unsigned nIteration; // current PatchMatch iteration
	int nIteration_external; // current PatchMatch iteration
	const CLISTDEF0IDX(ViewData,IIndex) images; // neighbor images used
	const DepthData::ViewData& image0;
	const DepthData::ViewData& image1;

	#if DENSE_NCC != DENSE_NCC_WEIGHTED
	const Image64F& image0Sum; // integral image used to fast compute patch mean intensity
	#endif
	const MapRefArr& coords;
	const Image8U::Size size;
	const Depth dMin, dMax;
	float& delta_c2pmax;  
	const Depth dMinSqr, dMaxSqr;
	const ENDIRECTION dir;
	#if DENSE_AGGNCC == DENSE_AGGNCC_NTH || DENSE_AGGNCC == DENSE_AGGNCC_MINMEAN
	const IDX idxScore;
	#endif

	DepthEstimator(
		unsigned nIter, int nIter_external, DepthData& _depthData0, DepthDataArr& _arrDepthData0, volatile Thread::safe_t& _idx,
		#if DENSE_NCC == DENSE_NCC_WEIGHTED
		WeightMap& _weightMap0,
		#else
		const Image64F& _image0Sum,
		#endif
		const MapRefArr& _coords);

	bool PreparePixelPatch(const ImageRef&);
	bool FillPixelPatch();
	float ScorePixelImage(int idxView, const ViewData& image1, Depth, const Normal&);
	float ScorePixel(Depth, const Normal&);
	void ProcessPixel(IDX idx);
	Depth InterpolatePixel(const ImageRef&, Depth, const Normal&) const;
	#if DENSE_SMOOTHNESS == DENSE_SMOOTHNESS_PLANE
	void InitPlane(Depth, const Normal&);
	#endif
	#if DENSE_REFINE == DENSE_REFINE_EXACT
	PixelEstimate PerturbEstimate(const PixelEstimate&, float perturbation);
	#endif

	#if DENSE_NCC != DENSE_NCC_WEIGHTED
	inline float GetImage0Sum(const ImageRef& p) const {
		const ImageRef p0(p.x-nSizeHalfWindow, p.y-nSizeHalfWindow);
		const ImageRef p1(p0.x+nSizeWindow, p0.y);
		const ImageRef p2(p0.x, p0.y+nSizeWindow);
		const ImageRef p3(p1.x, p2.y);
		return (float)(image0Sum(p3) - image0Sum(p2) - image0Sum(p1) + image0Sum(p0));
	}
	#endif

	#if DENSE_NCC == DENSE_NCC_WEIGHTED
	float GetWeight(const ImageRef& x, float center) const {
		// color weight [0..1]
		const float sigmaColor(-1.f/(2.f*SQUARE(0.2f)));
		const float wColor(SQUARE(image0.image(x0+x)-center) * sigmaColor);
		// spatial weight [0..1]
		
		// const float sigmaSpatial(-1.f/(2.f*SQUARE((int)nSizeHalfWindow)));
		const float sigmaSpatial(-1.f/(2.f*SQUARE((int)adapthalfwin)));		

		const float wSpatial(float(SQUARE(x.x) + SQUARE(x.y)) * sigmaSpatial);
		return DENSE_EXP(wColor+wSpatial);
	}

	
	float GetWeight_part(const ImageRef& x, float center) const {
		// color weight [0..1]
		const float sigmaColor(-1.f/(2.f*SQUARE(0.2f)));
		const float wColor(SQUARE(image0.image(x0+x)-center) * sigmaColor);
		// spatial weight [0..1]
		const float sigmaSpatial(-1.f/2.f);
		const float wSpatial(float(SQUARE(x.x) + SQUARE(x.y)) * sigmaSpatial);
		return DENSE_EXP(wColor+wSpatial);
	}
	#endif

	
	inline Matrix3x3f ComputeHomographyMatrix(const ViewData& img, Depth depth, const Normal& normal) const {
		#if 0
		// compute homography matrix
		const Matrix3x3f H(img.view.camera.K*HomographyMatrixComposition(image0.camera, img.view.camera, Vec3(normal), Vec3(X0*depth))*image0.camera.K.inv());
		#else
		// compute homography matrix as above, caching some constants
		const Vec3 n(normal);
		return (img.Hl + img.Hm * (n.t()*INVERT(n.dot(X0)*depth))) * img.Hr;
		#endif
	}
	
	inline Matrix3x3f ComputeHomoMatrix(const ViewData& img, Depth depth, const Normal& normal,ImageRef x1) const {
		#if 0
		// compute homography matrix
		const Matrix3x3f H(img.view.camera.K*HomographyMatrixComposition(image0.camera, img.view.camera, Vec3(normal), Vec3(X0*depth))*image0.camera.K.inv());
		#else
		// compute homography matrix as above, caching some constants
		// Hij=Kj*(Rj*inv(Ri)+(Rj*(Ci-Cj)*ni)/(ni*Xi))*inv(Ki)
		const Vec3 n(normal);
		Vec3 X1 = img.view.camera.TransformPointI2C(Cast<REAL>(x1));
		return (img.Hl1 + img.Hm1 * (n.t()*INVERT(n.dot(X1)*depth))) * img.Hr1;
		#endif
	}

	inline Matrix3x3f ComputeFundamentalMatrix(const ViewData& img) const {
		#if 0
		// compute homography matrix
		const Matrix3x3f H(img.view.camera.K*HomographyMatrixComposition(image0.camera, img.view.camera, Vec3(normal), Vec3(X0*depth))*image0.camera.K.inv());
		#else
		return img.Hr1.t()*img.E*img.Hr1;
		#endif
	}
	static inline CLISTDEF0IDX(ViewData,IIndex) InitImages(const DepthData& depthData) {
		CLISTDEF0IDX(ViewData,IIndex) images(0, depthData.images.GetSize()-1);
		const DepthData::ViewData& image0(depthData.images.First());
		for (IIndex i=1; i<depthData.images.GetSize(); ++i)
			images.AddConstruct(image0, depthData.images[i]);
		return images;
	}

	static inline Point3 ComputeRelativeC(const DepthData& depthData) {
		return depthData.images[1].camera.R*(depthData.images[0].camera.C-depthData.images[1].camera.C);
	}
	static inline Matrix3x3 ComputeRelativeR(const DepthData& depthData) {
		RMatrix R;
		ComputeRelativeRotation(depthData.images[0].camera.R, depthData.images[1].camera.R, R);
		return R;
	}

	// generate random depth and normal
	inline Depth RandomDepth(Depth dMinSqr, Depth dMaxSqr) {
		ASSERT(dMinSqr > 0 && dMinSqr < dMaxSqr);
		return SQUARE(rnd.randomRange(dMinSqr, dMaxSqr));
	}
	inline Normal RandomNormal(const Point3f& viewRay) {
		Normal normal;
		Dir2Normal(Point2f(rnd.randomRange(FD2R(0.f),FD2R(180.f)), rnd.randomRange(FD2R(90.f),FD2R(180.f))), normal);
		return normal.dot(viewRay) > 0 ? -normal : normal;
	}

	// adjust normal such that it makes at most 90 degrees with the viewing angle
	inline void CorrectNormal(Normal& normal) const {
		const Normal viewDir(Cast<float>(X0));
		const float cosAngLen(normal.dot(viewDir));
		if (cosAngLen >= 0)
			normal = RMatrixBaseF(normal.cross(viewDir), MINF((ACOS(cosAngLen/norm(viewDir))-FD2R(90.f))*1.01f, -0.001f)) * normal;
	}

	static bool ImportIgnoreMask(const Image&, const Image8U::Size&, BitMatrix&, String& nIgnoreMaskLabel);
	static void MapMatrix2ZigzagIdx(const Image8U::Size& size, DepthEstimator::MapRefArr& coords, const BitMatrix& mask, int rawStride=16);

	const float smoothBonusDepth, smoothBonusNormal;
	const float smoothSigmaDepth, smoothSigmaNormal;
	const float thMagnitudeSq;
	const float angle1Range, angle2Range;
	const float thConfSmall, thConfBig, thConfRand;
	const float thRobust;
	#if DENSE_REFINE == DENSE_REFINE_EXACT
	const float thPerturbation;
	#endif
	static const float scaleRanges[12];
};
/*----------------------------------------------------------------*/


// Tools
bool TriangulatePoints2DepthMap(
	const DepthData::ViewData& image, const PointCloud& pointcloud, const IndexArr& points,
	DepthMap& depthMap, NormalMap& normalMap, Depth& dMin, Depth& dMax);
bool TriangulatePoints2DepthMap(
	const DepthData::ViewData& image, const PointCloud& pointcloud, const IndexArr& points,
	DepthMap& depthMap, Depth& dMin, Depth& dMax);

MVS_API unsigned EstimatePlane(const Point3Arr&, Plane&, double& maxThreshold, bool arrInliers[]=NULL, size_t maxIters=0);
MVS_API unsigned EstimatePlaneLockFirstPoint(const Point3Arr&, Plane&, double& maxThreshold, bool arrInliers[]=NULL, size_t maxIters=0);
MVS_API unsigned EstimatePlaneTh(const Point3Arr&, Plane&, double maxThreshold, bool arrInliers[]=NULL, size_t maxIters=0);
MVS_API unsigned EstimatePlaneThLockFirstPoint(const Point3Arr&, Plane&, double maxThreshold, bool arrInliers[]=NULL, size_t maxIters=0);

MVS_API void EstimatePointColors(const ImageArr& images, PointCloud& pointcloud);
MVS_API void EstimatePointLabels(const ImageArr& images, PointCloud& pointcloud);
MVS_API void EstimatePointNormals(const ImageArr& images, PointCloud& pointcloud, int numNeighbors=16/*K-nearest neighbors*/);

MVS_API bool EstimateNormalMap(const Matrix3x3f& K, const DepthMap&, NormalMap&);

MVS_API bool SaveDepthMap(const String& fileName, const DepthMap& depthMap);
MVS_API bool LoadDepthMap(const String& fileName, DepthMap& depthMap);
MVS_API bool SaveNormalMap(const String& fileName, const NormalMap& normalMap);
MVS_API bool LoadNormalMap(const String& fileName, NormalMap& normalMap);
MVS_API bool SaveConfidenceMap(const String& fileName, const ConfidenceMap& confMap);
MVS_API bool LoadConfidenceMap(const String& fileName, ConfidenceMap& confMap);

MVS_API Image8U3 DepthMap2Image(const DepthMap& depthMap, Depth minDepth=FLT_MAX, Depth maxDepth=0);
MVS_API bool ExportDepthMap(const String& fileName, const DepthMap& depthMap, Depth minDepth=FLT_MAX, Depth maxDepth=0);
MVS_API bool ExportDepthMapByJetColormap(const String& fileName, const DepthMap& depthMap); 
MVS_API bool ExportDepthDiffmap(const String& fileName, const DepthMap& depthMap); 
MVS_API bool ExportSuperPixelLabelsmap(const String& fileName, const Image16I& labelsMap); 
MVS_API bool ExportBinDepth(DepthData& depthData);      
MVS_API bool ExportNormalMap(const String& fileName, const NormalMap& normalMap);
MVS_API bool ExportConfidenceMap(const String& fileName, const ConfidenceMap& confMap);
MVS_API bool ExportPointCloud(const String& fileName, const Image&, const DepthMap&, const NormalMap&);

MVS_API bool ExportDepthDataRaw(const String&, const String& imageFileName,
	const IIndexArr&, const cv::Size& imageSize,
	const KMatrix&, const RMatrix&, const CMatrix&,
	Depth dMin, Depth dMax,
	const DepthMap&, const NormalMap&, const ConfidenceMap&);
MVS_API bool ImportDepthDataRaw(const String&, String& imageFileName,
	IIndexArr&, cv::Size& imageSize,
	KMatrix&, RMatrix&, CMatrix&,
	Depth& dMin, Depth& dMax,
	DepthMap&, NormalMap&, ConfidenceMap&, unsigned flags=7);

MVS_API void CompareDepthMaps(const DepthMap& depthMap, const DepthMap& depthMapGT, uint32_t idxImage, float threshold=0.01f);
MVS_API void CompareNormalMaps(const NormalMap& normalMap, const NormalMap& normalMapGT, uint32_t idxImage);
/*----------------------------------------------------------------*/


// Jet colormap inspired by Matlab. Grayvalues are expected in the range [0, 1]
// and are converted to RGB values in the same range.
class JetColormap {
 public:
  static float Red(const float gray);
  static float Green(const float gray);
  static float Blue(const float gray);

 private:
  static float Interpolate(const float val, const float y0, const float x0,
                           const float y1, const float x1);
  static float Base(const float val);
};



} // namespace MVS

#endif // _MVS_DEPTHMAP_H_
