/*
 *	This file is part of qpOASES.
 *
 *	qpOASES -- An Implementation of the Online Active Set Strategy.
 *	Copyright (C) 2007-2017 by Hans Joachim Ferreau, Andreas Potschka,
 *	Christian Kirches et al. All rights reserved.
 *
 *	qpOASES is free software; you can redistribute it and/or
 *	modify it under the terms of the GNU Lesser General Public
 *	License as published by the Free Software Foundation; either
 *	version 2.1 of the License, or (at your option) any later version.
 *
 *	qpOASES is distributed in the hope that it will be useful,
 *	but WITHOUT ANY WARRANTY; without even the implied warranty of
 *	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *	See the GNU Lesser General Public License for more details.
 *
 *	You should have received a copy of the GNU Lesser General Public
 *	License along with qpOASES; if not, write to the Free Software
 *	Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
 *
 */

/**
 *	\file examples/example1.cpp
 *	\author Hans Joachim Ferreau
 *	\version 3.2
 *	\date 2007-2017
 *
 *	Very simple example for testing qpOASES using the QProblem class.
 */

#define GLOG_USE_GLOG_EXPORT

#include <glog/logging.h>
#include <Eigen/Dense>
#include <iostream>
#include <qpOASES.hpp>

/** Example for qpOASES main function using the QProblem class. */
// int main() {
//   //   USING_NAMESPACE_QPOASES

//   // Using Eigen library
//   Eigen::Matrix2d H{{1.0, 0.0}, {0.0, 0.5}};
//   Eigen::RowVector2d A{1.0, 1.0};
//   Eigen::Vector2d g{1.5, 1.0};
//   Eigen::Vector2d lb{0.5, -2.0};
//   Eigen::Vector2d ub{5.0, 2.0};
//   Eigen::Vector<double, 1> lbA{-1.0};
//   Eigen::Vector<double, 1> ubA{2.0};

//   /* Setup data of second QP. */
//   qpOASES::real_t g_new[2] = {1.0, 1.5};
//   qpOASES::real_t lb_new[2] = {0.0, -1.0};
//   qpOASES::real_t ub_new[2] = {5.0, -0.5};
//   qpOASES::real_t lbA_new[1] = {-2.0};
//   qpOASES::real_t ubA_new[1] = {1.0};

//   /* Setting up QProblem object. */
//   qpOASES::QProblem example(2, 1);

//   qpOASES::Options options;

//   example.setOptions(options);

//   /* Solve first QP. */
//   qpOASES::int_t nWSR = 100;
//   example.init(H.data(), g.data(), A.data(), lb.data(), ub.data(), lbA.data(), ubA.data(), nWSR);

//   /* Get and print solution of first QP. */
//   qpOASES::real_t xOpt[2];
//   qpOASES::real_t yOpt[2 + 1];
//   example.getPrimalSolution(xOpt);
//   example.getDualSolution(yOpt);
//   printf("\nxOpt = [ %e, %e ];  yOpt = [ %e, %e, %e ];  objVal = %e\n\n", xOpt[0], xOpt[1], yOpt[0], yOpt[1],
//   yOpt[2],
//          example.getObjVal());

//   /* Solve second QP. */
//   nWSR = 10;
//   example.hotstart(g_new, lb_new, ub_new, lbA_new, ubA_new, nWSR);

//   /* Get and print solution of second QP. */
//   example.getPrimalSolution(xOpt);
//   example.getDualSolution(yOpt);
//   printf("\nxOpt = [ %e, %e ];  yOpt = [ %e, %e, %e ];  objVal = %e\n\n", xOpt[0], xOpt[1], yOpt[0], yOpt[1],
//   yOpt[2],
//          example.getObjVal());

//   example.printOptions();

//   std::cout << "Iterations taken: " << nWSR << std::endl;
//   std::cout << "Final termination tolerance (precision): " << options.terminationTolerance << std::endl;
//   /*example.printProperties();*/

//   /*getGlobalMessageHandler()->listAllMessages();*/

//   return 0;
// }

int main() {
  // google::InitGoogleLogging("controller");
  FLAGS_logtostderr = 1;
  FLAGS_colorlogtostderr = 1;

  qpOASES::real_t H[1] = {2.0};
  qpOASES::real_t g[1] = {0};
  qpOASES::real_t A[1] = {1};
  qpOASES::real_t lbA[1] = {2};
  qpOASES::real_t ubA[1] = {4};

  qpOASES::QProblem example(1, 1);
  qpOASES::Options options;
  // options.printLevel = qpOASES::PL_NONE;
  options.terminationTolerance = 1e-6;
  options.setToMPC();
  example.setOptions(options);

  /* Solve first QP. */
  qpOASES::int_t nWSR = 100;
  qpOASES::returnValue qp_returnvalue;
  // qp_returnvalue = example.init(H, g, nullptr, nullptr, nullptr, nullptr, nullptr, nWSR);
  qp_returnvalue = example.init(H, g, A, nullptr, nullptr, lbA, ubA, nWSR);
  qpOASES::real_t xOpt[1];
  qpOASES::real_t yOpt[1];
  example.getPrimalSolution(xOpt);
  example.getDualSolution(yOpt);
  example.printOptions();
  if (example.isSolved()) {
    std::cout << "Is Solved!" << std::endl;
  }
  printf("\nxOpt = [ %e ];  yOpt = [ %e ];  objVal = %e\n\n", xOpt[0], yOpt[0], example.getObjVal());
  std::cout << "Iterations taken: " << nWSR << std::endl;
  std::cout << "Final termination tolerance (precision): " << options.terminationTolerance << std::endl;

  //  Check qp_solver status
  int qp_status = qpOASES::getSimpleStatus(qp_returnvalue, qpOASES::BT_TRUE);
  switch (qp_status) {
    case 0:
      LOG(WARNING) << "QP problem solved";
      break;
    case 1:
      LOG(WARNING) << "QP could not be solved within given number of iterations";
      break;
    case -1:
      LOG(WARNING) << "QP could not be solved due to an internal error";
      break;
    case -2:
      LOG(WARNING) << "QP is infeasible (and thus could not be solved)";
      break;
    case -3:
      LOG(WARNING) << "QP is unbounded (and thus could not be solved)";
      break;

    default:
      break;
  }
  return 0;
}

/*
 *	end of file
 */
