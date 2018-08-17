#include <math.h>
#include "TSystem.h"
#include "TROOT.h"
#include "TRandom.h"
#include "TFile.h"
#include "FairPrimaryGenerator.h"
#include "MuonBackGenerator.h"
#include "TDatabasePDG.h"               // for TDatabasePDG
#include "TMath.h"                      // for Sqrt
#include "vetoPoint.h"                
#include "ShipMCTrack.h"                
#include "TMCProcess.h"
#include <algorithm>
#include <fstream>

// read events from Pythia8/Geant4 base simulation (only target + hadron absorber

// -----   Default constructor   -------------------------------------------
MuonBackGenerator::MuonBackGenerator() {}
// -------------------------------------------------------------------------
// -----   Default constructor   -------------------------------------------
Bool_t MuonBackGenerator::Init(const char* fileName) {
  return Init(fileName, 0, false);
}
// -----   Default constructor   -------------------------------------------
Bool_t MuonBackGenerator::Init(const char* fileName, const int firstEvent, const Bool_t fl = false ) {
  fLogger = FairLogger::GetLogger();
  fLogger->Info(MESSAGE_ORIGIN,"Opening input file %s",fileName);
  if (0 == strncmp("/eos",fileName,4) ) {
     TString tmp = gSystem->Getenv("EOSSHIP");
     tmp+=fileName;
     fInputFile  = TFile::Open(tmp); 
  }else{
  fInputFile  = new TFile(fileName);
  }
  if (fInputFile->IsZombie()) {
    fLogger->Fatal(MESSAGE_ORIGIN, "Error opening the Signal file:",fInputFile);
  }
  fn = firstEvent;
  fPhiRandomize = fl;
  fSameSeed = 0;
  fsmearBeam = 0; // default no beam smearing, use SetSmearBeam(sb) if different, sb [cm]
  fTree = (TTree *)fInputFile->Get("pythia8-Geant4");
  if (fTree){
   fNevents = fTree->GetEntries();
  // count only events with muons
  // fMuons  = fTree->Draw("id","abs(id)==13","goff");
   fTree->SetBranchAddress("id",&id);                // particle id
   fTree->SetBranchAddress("parentid",&parentid);    // parent id, could be different
   fTree->SetBranchAddress("pythiaid",&pythiaid);    // pythiaid original particle
   fTree->SetBranchAddress("ecut",&ecut);    // energy cut used in simulation
   fTree->SetBranchAddress("w",&w);                  // weight of event
//  check if ntuple has information of momentum at origin
   if (fTree->GetListOfLeaves()->GetSize() < 17){  
    fTree->SetBranchAddress("x",&vx);   // position with respect to startOfTarget at -89.27m
    fTree->SetBranchAddress("y",&vy);
    fTree->SetBranchAddress("z",&vz);
    fTree->SetBranchAddress("px",&px);   // momentum
    fTree->SetBranchAddress("py",&py);
    fTree->SetBranchAddress("pz",&pz);
   }else{
    fTree->SetBranchAddress("ox",&vx);   // position with respect to startOfTarget at -50m
    fTree->SetBranchAddress("oy",&vy);
    fTree->SetBranchAddress("oz",&vz);
    fTree->SetBranchAddress("opx",&px);   // momentum
    fTree->SetBranchAddress("opy",&py);
    fTree->SetBranchAddress("opz",&pz);
   }
  }else{
   id = -1;
   fTree = (TTree *)fInputFile->Get("cbmsim");
   fNevents   = fTree->GetEntries();

   std::cout << "fNevents " << fNevents <<"\n";


   MCTrack = new TClonesArray("ShipMCTrack");
   vetoPoints = new TClonesArray("vetoPoint");
   fTree->SetBranchAddress("MCTrack",&MCTrack);
   fTree->SetBranchAddress("vetoPoint",&vetoPoints);
  }    
  return kTRUE;
}
// -----   Destructor   ----------------------------------------------------
MuonBackGenerator::~MuonBackGenerator()
{
}
// -------------------------------------------------------------------------

// -----   Passing the event   ---------------------------------------------
Bool_t MuonBackGenerator::ReadEvent(FairPrimaryGenerator* cpg)
{
TDatabasePDG* pdgBase = TDatabasePDG::Instance();

Double_t mass,e,tof,phi;
Double_t dx = 0, dy = 0;

std::vector<int> muList;

//define variables to pull from tree
Double_t id2,parentid2,pythiaid2,ecut2,w2,x2,y2,z2,px2,py2,pz2,release_time;
Double_t fUniquieID,fPdgCode,fMotherId,fPx,fPy,fPz,fM,fStartX,fStartY,fStartZ,fW,fProcID;
// Float_t id2,parentid2,pythiaid2,ecut2,w2,x2,y2,z2,px2,py2,pz2,release_time;

while (fn<fNevents) 
{

  if (fn>fNevents)
  {
    std::cout << "3 " << fn <<"\n";
    fLogger->Info(MESSAGE_ORIGIN,"End of file reached %i",fNevents);
    return kFALSE;
  } 

  fTree->SetBranchAddress("fUniquieID",&fUniquieID);
  fTree->SetBranchAddress("fPdgCode",&fPdgCode);
  fTree->SetBranchAddress("fMotherId",&fMotherId);
  fTree->SetBranchAddress("fPx",&fPx);
  fTree->SetBranchAddress("fPy",&fPy);
  fTree->SetBranchAddress("fPz",&fPz);
  fTree->SetBranchAddress("fM",&fM);
  fTree->SetBranchAddress("fStartX",&fStartX);
  fTree->SetBranchAddress("fStartY",&fStartY);
  fTree->SetBranchAddress("fStartZ",&fStartZ);
  fTree->SetBranchAddress("fW",&fW);
  fTree->SetBranchAddress("fProcID",&fProcID);



  fTree->GetEntry(fn);

  // if (fn % 1000 == 0)
  // {
    std::cout << fn <<"\n";
  // } 
  //add track to simulation

  // cpg->AddTrack(13,px2,py2,pz2,x2,y2,z2+2084.5,-1,true,ecut2,0,1);
  // cpg->AddTrack(13,px2,py2,pz2,x2,y2,z2,-1,true,ecut2,0,1);
  // std::cout << 13 <<" "<<px2<<" "<<py2<<" "<<pz2<<" "<<x2<<" "<<y2<<" "<<-6542<<" "<<-1<<" "<<true<<" "<<ecut2<<" "<<0<<" "<<1 <<"\n";
  cpg->AddTrack(fPdgCode,fPx,fPy,fPz,fStartX,fStartY,fStartZ,-1,true,ecut2,0,fW);

  fn++;

  return kTRUE;
}
}

// -------------------------------------------------------------------------
Int_t MuonBackGenerator::GetNevents()
{
 return fNevents;
}
void MuonBackGenerator::CloseFile()
{
 fInputFile->Close();
 fInputFile->Delete();
 delete fInputFile;
}

ClassImp(MuonBackGenerator)

