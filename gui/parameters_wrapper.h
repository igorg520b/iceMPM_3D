#ifndef PARAMETERS_WRAPPER_H
#define PARAMETERS_WRAPPER_H


#include <QObject>
#include <QString>
#include "simulation/parameters_sim_3d.h"
#include <cmath>

// wrapper for SimParams to display/edit them in GUI
class ParamsWrapper : public QObject
{
    Q_OBJECT

    icy::SimParams3D *prms;

    // InitialTimeStep
    Q_PROPERTY(double in_TimeStep READ getTimeStep WRITE setTimeStep NOTIFY propertyChanged)
    double getTimeStep() {return prms->InitialTimeStep;}
    void setTimeStep(double val) { prms->InitialTimeStep = val; prms->ComputeHelperVariables();}

    Q_PROPERTY(QString in_TimeStep_ READ getTimeStep_ NOTIFY propertyChanged)
    QString getTimeStep_() {return QString("%1 s").arg(prms->InitialTimeStep,0,'e',1);}


    // SimulationEndTime
    Q_PROPERTY(double in_SimulationTime READ getSimulationTime WRITE setSimulationTime NOTIFY propertyChanged)
    double getSimulationTime() {return prms->SimulationEndTime;}
    void setSimulationTime(double val) { prms->SimulationEndTime = val; }

    // UpdateEveryNthStep
    Q_PROPERTY(int in_UpdateEvery READ getUpdateEveryNthStep NOTIFY propertyChanged)
    int getUpdateEveryNthStep() {return prms->UpdateEveryNthStep;}

    // YoungsModulus
    Q_PROPERTY(double p_YoungsModulus READ getYoungsModulus WRITE setYoungsModulus NOTIFY propertyChanged)
    double getYoungsModulus() {return prms->YoungsModulus;}
    void setYoungsModulus(double val) { prms->YoungsModulus = (float)val; prms->ComputeLame(); }

    Q_PROPERTY(QString p_YM READ getYM NOTIFY propertyChanged)
    QString getYM() {return QString("%1 Pa").arg(prms->YoungsModulus, 0, 'e', 2);}


    // PoissonsRatio
    Q_PROPERTY(double p_PoissonsRatio READ getPoissonsRatio WRITE setPoissonsRatio NOTIFY propertyChanged)
    double getPoissonsRatio() {return prms->PoissonsRatio;}
    void setPoissonsRatio(double val) { prms->PoissonsRatio = (float)val; prms->ComputeLame(); }

    Q_PROPERTY(double p_LameMu READ getMu NOTIFY propertyChanged)
    double getMu() {return prms->mu;}

    Q_PROPERTY(double p_LameKappa READ getKappa NOTIFY propertyChanged)
    double getKappa() {return prms->kappa;}


    // ParticleViewSize
    Q_PROPERTY(double p_ParticleViewSize READ getParticleViewSize WRITE setParticleViewSize NOTIFY propertyChanged)
    double getParticleViewSize() {return prms->ParticleViewSize;}
    void setParticleViewSize(double val) {prms->ParticleViewSize=val;}

    // indenter
    Q_PROPERTY(double IndDiameter READ getIndDiameter NOTIFY propertyChanged)
    double getIndDiameter() {return prms->IndDiameter;}

    Q_PROPERTY(double IndVelocity READ getIndVelocity WRITE setIndVelocity NOTIFY propertyChanged)
    double getIndVelocity() {return prms->IndVelocity;}
    void setIndVelocity(double val) {prms->IndVelocity=val;}

    Q_PROPERTY(double IndDepth READ getIndDepth NOTIFY propertyChanged)
    double getIndDepth() {return prms->IndDepth;}

    Q_PROPERTY(double IndAngSubd READ getIndAngSubd NOTIFY propertyChanged)
    double getIndAngSubd() {return prms->n_indenter_subdivisions_angular;}

    Q_PROPERTY(double IndArraySize READ getIndArraySize NOTIFY propertyChanged)
    double getIndArraySize() {return prms->indenter_array_size;}


    // ice block
    Q_PROPERTY(int PtActual READ getPointCountActual NOTIFY propertyChanged)
    int getPointCountActual() {return prms->nPts;}

    Q_PROPERTY(double PtsPerCell READ getPointsPerCell NOTIFY propertyChanged)
    double getPointsPerCell() {return prms->PointsPerCell();}


    Q_PROPERTY(int PtPitch READ getPointPitch NOTIFY propertyChanged)
    int getPointPitch() {return prms->nPtsPitch;}

    Q_PROPERTY(QString Grid READ getGridDimensions NOTIFY propertyChanged)
    QString getGridDimensions() {return QString("%1 x %2 x %3").arg(prms->GridX).arg(prms->GridY).arg(prms->GridZ);}

    Q_PROPERTY(double nacc_beta READ getNaccBeta NOTIFY propertyChanged)
    double getNaccBeta() {return prms->NACC_beta;}

    Q_PROPERTY(double nacc_pc READ getNaccPc NOTIFY propertyChanged)
    double getNaccPc() {return (1 - prms->NACC_beta)*prms->IceCompressiveStrength/2.;}

    Q_PROPERTY(double nacc_M READ getNaccM NOTIFY propertyChanged)
    double getNaccM() {return sqrt(prms->NACC_M);}


    // Drucker-Prager
    Q_PROPERTY(double DP_phi READ getDPPhi WRITE setDPPhi NOTIFY propertyChanged)
    double getDPPhi() {return std::atan(prms->DP_tan_phi)*180/icy::SimParams3D::pi;}
    void setDPPhi(double val) {prms->DP_tan_phi = tan(val*icy::SimParams3D::pi/180);}

    Q_PROPERTY(double DP_tan_phi READ getDPTanPhi NOTIFY propertyChanged)
    double getDPTanPhi() {return prms->DP_tan_phi;}


    // Failure surface
    Q_PROPERTY(double ice_CompressiveStr READ getIce_CompressiveStr WRITE setIce_CompressiveStr NOTIFY propertyChanged)
    double getIce_CompressiveStr() {return prms->IceCompressiveStrength;}
    void setIce_CompressiveStr(double val) {prms->IceCompressiveStrength = val; prms->ComputeCamClayParams2();}

    Q_PROPERTY(double ice_TensileStr READ getIce_TensileStr WRITE setIce_TensileStr NOTIFY propertyChanged)
    double getIce_TensileStr() {return prms->IceTensileStrength;}
    void setIce_TensileStr(double val) {prms->IceTensileStrength = val; prms->ComputeCamClayParams2();}

    Q_PROPERTY(double ice_ShearStr READ getIce_ShearStr WRITE setIce_ShearStr NOTIFY propertyChanged)
    double getIce_ShearStr() {return prms->IceShearStrength;}
    void setIce_ShearStr(double val) {prms->IceShearStrength = val; prms->ComputeCamClayParams2();}


    // kernel execution
    Q_PROPERTY(int tpb_P2G READ get_tpb_P2G WRITE set_tpb_P2G NOTIFY propertyChanged)
    int get_tpb_P2G() {return prms->tpb_P2G;}
    void set_tpb_P2G(int val) { prms->tpb_P2G = val; }

    Q_PROPERTY(int tpb_Upd READ get_tpb_Upd WRITE set_tpb_Upd NOTIFY propertyChanged)
    int get_tpb_Upd() {return prms->tpb_Upd;}
    void set_tpb_Upd(int val) { prms->tpb_Upd = val; }

    Q_PROPERTY(int tpb_G2P READ get_tpb_G2P WRITE set_tpb_G2P NOTIFY propertyChanged)
    int get_tpb_G2P() {return prms->tpb_G2P;}
    void set_tpb_G2P(int val) { prms->tpb_G2P = val; }

    Q_PROPERTY(int SetupType READ get_SetupType NOTIFY propertyChanged)
    bool get_SetupType() {return prms->SetupType;}


public:
    ParamsWrapper(icy::SimParams3D *p)
    {
        this->prms = p;
    }


Q_SIGNALS:
    void propertyChanged();
};



#endif // PARAMETERS_WRAPPER_H
