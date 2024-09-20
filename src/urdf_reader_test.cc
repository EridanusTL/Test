#include <rbdl/rbdl.h>
#include <rbdl/rbdl_utils.h>

#include "third-party/rbdl/include/rbdl/addons/urdfreader/urdfreader.h"

#include <iomanip>
#include <iostream>

using namespace std;

bool verbose = true;
bool floatbase = true;
string filename = "/home/tlbot/code/Test/src/test.urdf";

using namespace std;

using namespace RigidBodyDynamics::Math;

int main(int argc, char* argv[]) {
  //   if (argc < 2) {
  //     cerr << "Error: not enough arguments!" << endl;
  //     usage(argv[0]);
  //   }

  bool verbose = false;
  bool dof_overview = true;
  bool model_hierarchy = true;
  bool body_origins = true;
  bool center_of_mass = true;

  //   string filename = argv[1];

  RigidBodyDynamics::Model model;

  if (!RigidBodyDynamics::Addons::URDFReadFromFile(filename.c_str(), &model, floatbase, verbose)) {
    cerr << "Loading of urdf model failed!" << endl;
    return -1;
  }

  cout << "Model loading successful!" << endl;

  if (dof_overview) {
    cout << "Degree of freedom overview:" << endl;
    cout << RigidBodyDynamics::Utils::GetModelDOFOverview(model);
  }

  if (model_hierarchy) {
    cout << "Model Hierarchy:" << endl;
    cout << RigidBodyDynamics::Utils::GetModelHierarchy(model);
  }

  if (body_origins) {
    cout << "Body Origins:" << endl;
    cout << RigidBodyDynamics::Utils::GetNamedBodyOriginsOverview(model);
  }

  if (center_of_mass) {
    VectorNd q_zero(VectorNd::Zero(model.q_size));
    VectorNd qdot_zero(VectorNd::Zero(model.qdot_size));
    RigidBodyDynamics::UpdateKinematics(model, q_zero, qdot_zero, qdot_zero);

    for (unsigned int i = 1; i < model.mBodies.size(); i++) {
      if (model.mBodies[i].mIsVirtual) continue;

      SpatialRigidBodyInertia rbi_base = model.X_base[i].apply(model.I[i]);
      Vector3d body_com = rbi_base.h / rbi_base.m;
      cout << setw(12) << model.GetBodyName(i) << ": " << setw(10) << body_com.transpose() << endl;
    }

    Vector3d model_com;
    double mass;
    RigidBodyDynamics::Utils::CalcCenterOfMass(model, q_zero, qdot_zero, NULL, mass, model_com);
    cout << setw(14) << "Model COM: " << setw(10) << model_com.transpose() << endl;
    cout << setw(14) << "Model mass: " << mass << endl;
  }

  return 0;
}