#include<algorithm>
#include<vector>
#include<cmath>

constexpr float gamma=1.5f;

enum class Boundary{
    Dirichlet,
    VonNeumann,
    Periodic
};

inline float max3(float a, float b, float c){return std::max(std::max(a,b),c);}


class HydroGrid {
public:

    
    HydroGrid(size_t n_,Boundary boundary_=Boundary::Periodic): n(n_), ncells(n_* n_), boundary(boundary_)
{
    data.resize(ncells * 12);

    rho = data.data();
    mx  = rho + ncells;
    my  = mx  + ncells;
    E   = my  + ncells;
    P= E+ncells;
    scratch0 = P  + ncells;

    scratch1 = scratch0 + ncells;
    scratch2 = scratch1 + ncells;
    scratch3 = scratch2 + ncells;
    scratch4 = scratch3 + ncells;
    


}


    inline size_t idx(size_t i, size_t j) const { return i * n + j; }

    inline float eos(float rho_,float internalEnergy);

    void updatePressure(float* energyDensity, float* P);

    void computePrimitives(float* primRho, float* vx, float* vy, float* internalEnergy);
    void computeVx(float* vx);
    void computeVy(float* vy);


    void computeXFlux(float* T0x, float* Txx, float* Txy, float* Tx0);
    void computeYFlux(float* T0y, float* Tyy, float* Txy, float* Ty0);
    float computeAlpha(float* lambdaP, float* lambdaM, float* vx, float* aPlus,float* aMinus,bool IsHorizontal);    
    inline float F_HLL(float ap,float am,float fl,float fr, float ul, float ur, float denom);

    void F_HLL_Vector(float* inputF, float* ap, float* am, float*outputData, bool isHorizontal);
    void clampPosDefQuants();

    Boundary boundary;
    float* rho;
    float* mx;
    float* my;
    float* E;
    float* P;


    float* scratch0;
    float* scratch1;
    float* scratch2;
    float* scratch3;
    float* scratch4;
    float* scratch5;
    float* scratch6;
    
    size_t ncells;
private:
    size_t n;
    std::vector<float> data;
};




float HydroGrid::eos(float rho_,float internalEnergy){
    return (gamma-1)*rho_*internalEnergy;
}

/*
 second argument set to  Pressure
*/
void HydroGrid::updatePressure(float* energyDensity, float* P){
    for(size_t i=0; i<ncells;++i){
        P[i]=eos(rho[i],energyDensity[i]);
    }
}

/*
sets:
arg1 = rho
arg2 = vx
arg3 = vy
arg4 = energy_density
*/
void HydroGrid::computePrimitives(float* primRho, float* vx, float* vy, float* energyDensity){

        float rhoAtI;
        float vSquared;
        for(size_t i=0; i<ncells; ++i){
            rhoAtI=std::max(.01f,rho[i]);
            
            primRho[i]=rhoAtI;
            vx[i]=mx[i]/rhoAtI;
            vy[i]=my[i]/rhoAtI;

            vSquared= vx[i]*vx[i]+ vy[i]*vy[i];
            energyDensity[i] = (E[i])/rhoAtI-.5f*vSquared;
        }

    }


/*sets input pointers to vx and vy*/
void HydroGrid::computeVx(float* vx){
    for(size_t i=0; i<ncells;++i){
        vx[i]=mx[i]/rho[i];
    }
}

void HydroGrid::computeVy(float* vy){
    for(size_t i=0; i<ncells;++i){
        vy[i]=my[i]/rho[i];
    }
}

void HydroGrid::clampPosDefQuants(){
    for(size_t i =0; i<ncells;++i){
        rho[i]=std::max(rho[i],.01f);
        P[i]=std::max(P[i],.01f);
    }
}

/*
sets:
arg1 = xMomentum   (Density flux thru x surface)
arg2 = rho*vx^2+P  (xMomentum flux thru ...)
arg3 = rho*vx*vy   (yMomentum flux thru ...)
arg4 = (E+P)*vx    (Energy flux thru ...)
*/  
void HydroGrid::computeXFlux(float* T0x, float* Txx, float* Txy, float* Tx0){

    for(size_t i=0; i<ncells;++i){

        T0x[i]=mx[i];

        Txx[i]=mx[i]*mx[i]/rho[i]+P[i];

        Txy[i]=mx[i]*my[i]/rho[i];

        Tx0[i]=mx[i]*(E[i]+P[i])/rho[i];

    }
}


/*
sets:
arg1 = yMomentum     (Density flux thru y )
arg2 = rho*vx*vy    (xMomentum flux thru y)
arg3 = rho*vy^2+P    (yMomentum flux thru y)
arg4 = (E+P)*vy      (Energy flux thru y)
*/  
void HydroGrid::computeYFlux(float* T0y, float* Txy, float* Tyy, float* Ty0)
    {

    for(size_t i=0; i<ncells;++i){

        T0y[i]=my[i];

    
        Txy[i]=mx[i]*my[i]/rho[i];

        Tyy[i]=my[i]*my[i]/rho[i]+P[i];

        Ty0[i]=my[i]*(E[i]+P[i])/rho[i];

    }
}


float HydroGrid::computeAlpha(float* lambdaP, float* lambdaM, float* v, float* aPlus,float* aMinus,bool isHorizontal){
        float cs;
        size_t indR;
        size_t indL;
        //index for left and right boundary
        
        size_t xShift =size_t(isHorizontal);
        size_t yShift =size_t(!isHorizontal);
    float maxAlpha=0.0f;



        for(size_t i=0; i<ncells; ++i){
            cs=std::sqrtf(gamma*P[i]/rho[i]);
            lambdaP[i]=v[i]+cs;
            lambdaM[i]=v[i]-cs;
        }

        
        //computing for n boundaries and n-1 cells-> boundaries go from -1/2 at i=0 to n-1/2 at i=n-1. interiors go from 0 at i=0 to n-2 at i=n-2
        //interior cell boundaries, i_th element is the left wall of i_th cell e.g. alpha[0] is right wall of 0
        //sweeping horizontally, we go full range of y's but only interior walls of x 
        // so from i=0 (y=1/2) to i=n-1 (y=n-1/2) but j=1 (x=1/2) to j=n-1 (x=n-2-1/2) which is the left wall of the n-2th cell at index n-2
        //sweeping vertically, we go full range of x's but only interior of y
        
        for(size_t i=yShift; i<n-yShift;++i){
            for(size_t j=xShift; j<n-xShift; ++j){
                indL=idx(i,j);
                indR=idx(i-yShift,j-xShift);
                aPlus[indL]  = max3(0.0f, lambdaP[indL], lambdaP[indR]);
                aMinus[indL] = max3(0.0f, -lambdaM[indL], -lambdaM[indR]);

                maxAlpha=max3(maxAlpha,aPlus[indL],aMinus[indL]);



            }
        }

        //we've computed alpha[1]->x=1/2 to alpha[n-2]->x=n-1-1/
        //we need to boundary check by computing at edges
        //sweeping horizontally, we run down the rows like so (i,0), (i,n-1) 
        //sweeping vertically, we run down the rows like so  (0,j), (n-1,j)


        if(boundary == Boundary::Periodic) {
            for(size_t i=0; i<n; ++i) {
                indR = idx(i*xShift,i*yShift);
                indL=  idx(i*xShift+(n-1)*yShift,i*yShift+(n-1)*xShift);




                aPlus[indL] = max3(lambdaP[indL], lambdaP[indR], 0.0f);
                aMinus[indL] = max3(-lambdaM[indL],-lambdaM[indR], 0.0f);
                
                aPlus[indR] = aPlus[indL];       
                aMinus[indR] = aMinus[indL];

                maxAlpha=max3(maxAlpha,aPlus[indL],aMinus[indL]);
                maxAlpha=max3(maxAlpha,aPlus[indR],aMinus[indR]);


            }
        }
            //von Neumann so we need an extra array of size N for x=-1/2 or left wall of cell 0

        if(boundary==Boundary::VonNeumann){
            for(size_t i=0; i<n; ++i){
                indL = idx(i*xShift,i*yShift);
                indR=  idx(i*xShift+(n-1)*yShift,i*yShift+(n-1)*xShift);

                aPlus[indL]=max3(lambdaP[indL],-lambdaM[indL],0.0f);
                aMinus[indL]=max3(-lambdaM[indL],lambdaP[indL],0.0f);

                aPlus[indR]=max3(lambdaP[indR],-lambdaM[indR],0.0f);
                aMinus[indR]=max3(-lambdaM[indR],lambdaP[indR],0.0f);

                maxAlpha=max3(maxAlpha,aPlus[indL],aMinus[indL]);
                maxAlpha=max3(maxAlpha,aPlus[indR],aMinus[indR]);


            }
        }

        if(boundary==Boundary::Dirichlet){
            for(size_t i=0; i<n; ++i){
                indL = idx(i*xShift,i*yShift);
                indR=  idx(i*xShift+(n-1)*yShift,i*yShift+(n-1)*xShift);

                aPlus[indL]=std::max(lambdaP[indL],0.0f);
                aMinus[indL]=std::max(-lambdaM[indL],0.0f);

                aPlus[indR]=std::max(lambdaP[indR],0.0f);
                aMinus[indR]=std::max(-lambdaM[indR],0.0f);

                maxAlpha=max3(maxAlpha,aPlus[indL],aMinus[indL]);
                maxAlpha=max3(maxAlpha,aPlus[indR],aMinus[indR]);
            }
        }

        return maxAlpha;


} 

inline float HydroGrid::F_HLL(float ap,float am,float fl,float fr, float ul, float ur, float denom){
    return (ap*fl+am*fr-ap*am*(ur-ul))/denom;
}


void HydroGrid::F_HLL_Vector(float* inputF, float* ap, float* am, float*outputData, bool isHorizontal){
    size_t indR;
    size_t indL;
    float denom;
    //integer indecies, range from 0 to n-2
    float* rhoFlux=inputF;
    float* mxFlux=inputF+ncells;
    float* myFlux= mxFlux+ncells;
    float* Eflux= myFlux+ncells;

    //half integer indecies, range from 0 to n-1
    float* rhoFHLL=outputData;
    float* mxFHLL=outputData+ncells;
    float* myFHLL= mxFHLL+ncells;
    float* EFHLL= myFHLL+ncells;


        
    size_t xShift =size_t(isHorizontal);
    size_t yShift =size_t(!isHorizontal);
    //interior walls
    for(size_t i=yShift; i<n-yShift;++i){
        for(size_t j=xShift; j<n-xShift; ++j){
            indL=idx(i,j);
            indR=idx(i-yShift,j-xShift);
            denom =std::max(ap[indL]+ap[indL],0.0f);
            rhoFHLL[indL]=F_HLL(ap[indL],am[indL],rhoFlux[indL],rhoFlux[indR],rho[indL],rho[indR],denom);

            mxFHLL[indL]=F_HLL(ap[indL],am[indL],mxFlux[indL],mxFlux[indR],mx[indL],mx[indR],denom);

            myFHLL[indL]=F_HLL(ap[indL],am[indL],myFlux[indL],myFlux[indR],my[indL],my[indR],denom);

            EFHLL[indL]=F_HLL(ap[indL],am[indL],Eflux[indL],Eflux[indR],E[indL],E[indR],denom);
        }
    } 
        float xFlip=(float(isHorizontal)-.5f)*2.0f;
        float yFlip=(float(!isHorizontal)-.5f)*2.0f;

        float rhoFluxGhost;
        float mxFluxGhost;
        float myFluxGhost;
        float EFluxGhost;

        if(boundary==Boundary::VonNeumann) {
            for(size_t i=0; i<n; ++i) {
                indR = idx(i*xShift,i*yShift);
                indL=  idx(i*xShift+(n-1)*yShift,i*yShift+(n-1)*xShift);
                denom =std::max(am[indL]+ap[indL],0.0f);

                //'ghost cell, pretending theres a left cell with negative normal velocity e.g for x wall
                // mx->-mx
                //rho->rho
                //P->P
                //my->my
                // rhoFlux=mx rho-> -mxrho
                // mxFlux=mx*mx+P-> (-mx)*(-mx)+P
                // myFlux=mx*my*rho->-mx*my*rho
                // Eflux->mx(E+P)->-mx(E+P)
                rhoFluxGhost=-rhoFlux[indL];
                


                rhoFHLL[indL]=F_HLL(ap[indL],am[indL],rhoFluxGhost[indL], rhoFlux[indL], rho[indL],rho[indL],denom);
                mxFHLL[indL]= F_HLL(ap[indL],am[indL],mxFlux[indL] , mxFlux[indL],  -mx[indL], mx[indL], denom);
                myFHLL[indL]= F_HLL(ap[indL],am[indL],-myFlux[indL] , myFlux[indL],  my[indL], my[indL], denom);
                EFHLL[indL]=  F_HLL(ap[indL],am[indL],-Eflux[indL]  , Eflux[indL],   E[indL],  E[indL],  denom);



                rhoFHLL[indR]=F_HLL(ap[indR],am[indR],-rhoFlux[indR], rhoFlux[indR], rho[indR],rho[indR],denom);
                mxFHLL[indR]= F_HLL(ap[indR],am[indR],mxFlux[indR] , mxFlux[indR],  -mx[indR], mx[indR], denom);
                myFHLL[indR]= F_HLL(ap[indR],am[indR],-myFlux[indR] , myFlux[indR],  my[indR], my[indR], denom);
                EFHLL[indR]=  F_HLL(ap[indR],am[indR],-Eflux[indR]  , Eflux[indR],   E[indR],  E[indR],  denom);
            }
        }






}
