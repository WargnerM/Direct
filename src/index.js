'use strict';
const fs = require('fs');
const path = require('path');
const Matrix = require('ml-matrix').Matrix;
const isAnyArray= require('is-any-array');
const conhull = require('./util/conhull');

/**
 * Creates new PCA (Principal Component Analysis) from the dataset
 * @param {Matrix} fun - dataset or covariance matrix.
 * @param {Array} xU - dataset or covariance matrix.
 * @param {Array} xL - dataset or covariance matrix.
 * @param {Object} [options]
 * @param {number} [options.printLevel] - Parameter to show information.
 * @param {Object} [options.initialState] - Parameter with information.
 * */

function Direct(fun, xL, xU, options = {}, initialState = {}) {
  const opts = {
    iterations: 50,
    epsilon: 1e-4,
    tol: 0.01,
  };

  options = Object.assign({}, opts, options);

  if (fun === undefined || xL === undefined || xU === undefined) {
    throw new Error('There is something undefined');
  }

  if (isAnyArray(xL)) {
    xL = new Float32Array(xL);
  }

  if (isAnyArray(xU)) {
    xU = new Float32Array(xU);
  }

  if (xL.length !== xU.length) {
    throw new Error(
      'Lower bounds and Upper bounds for x are not of the same length',
    );
  }

  //-------------------------------------------------------------------------
  //                        STEP 1. Initialization
  //-------------------------------------------------------------------------
  let iterations = options.iterations;
  let epsilon = options.epsilon;
  let tol = options.tol;
  let funCalls = 0;
  let n = xL.length;
  let tolle = 1e-16;
  let tolle2 = 1e-6;
  let dMin = initialState.dMin;
  let diffBorders = xU.map((x, i) => x - xL[i]);

  let F, m, D, L, d, fMin, E, iMin, C;
  if (initialState.C && initialState.C.length > 0) {

    console.log('entra initial');
    F = initialState.F;
    m = F.length - 1;
    D = initialState.D;
    L = initialState.L;
    d = initialState.d;
    dMin = initialState.dMin;
    funCalls = initialState.funCalls;

    fMin = getMinValue(F);
    E = epsilon * Math.abs(fMin) > 1e-8 ? epsilon * Math.abs(fMin) : 1e-8;
    iMin = getIndexOfMin(F, D, E, fMin);

    C = initialState.C.slice();
    for (let j = 0; j < F.length; j++) {
      for (let i = 0; i < xL.length; i++) {
        C[j][i] = (C[j][i] - xL[i]) / diffBorders[i];
      }
    }
  } else {
    m = 0;
    C = [new Float32Array(n).fill(0.5)];
    let xM = new Float32Array(n);
    for (let i = 0; i < xL.length; i++) {
      xM[i] = xL[i] + C[0][i] * diffBorders[i];
    }
    fMin = fun(xM);
    funCalls = funCalls + 1;
    iMin = 0;
    L = [new Float32Array(n).fill(0.5)];
    D = [Math.sqrt(n * Math.pow(0.5, 2))];
    F = [fMin];
    d = D;
    dMin = [fMin];
  }

  let t = 0;
  //-------------------------------------------------------------------------
  //                          Iteration loop
  //-------------------------------------------------------------------------

  while (t < options.iterations) {
    //----------------------------------------------------------------------
    //  STEP 2. Identify the set S of all potentially optimal rectangles
    //----------------------------------------------------------------------
    let S1 = new Uint32Array(F.length);
    let idx = d.findIndex((e) => e === D[iMin]);
    let last = 0;
    for (let i = idx; i < d.length; i++) {
      for (let f = 0; f < F.length; f++) {
        if (F[f] === dMin[i]) {
          if (D[f] === d[i]) {
            S1[last++] = f;
          }
        }
      }
    }
    let S, S3;
    if (d.length - idx > 2) {
      let a1 = D[iMin];
      let b1 = F[iMin];
      let a2 = d[d.length - 1];
      let b2 = dMin[d.length - 1];
      let slope = (b2 - b1) / (a2 - a1);
      let constant = b1 - slope * a1;
      let ff = [];
      let dd = [];
      let S2 = new Uint32Array(last);
      last = 0;
      for (let i = 0; i < S2.length; i++) {
        let j = S1[i];
        if (F[j] <= slope * D[j] + constant + tolle2) {
          S2[last++] = j;
        }
      }
      let xx = new Array(last);
      let yy = new Array(last);
      for (let i = 0; i < xx.length; i++) {
        xx[i] = [D[S2[i]]];
        yy[i] = [F[S2[i]]];
      }
      let h = conhull(xx, yy);
      S3 = new Uint32Array(h.length);
      for (let i = 0; i < h.length; i++) {
        S3[i] = S2[h[i]];
      }
    } else {
      S3 = S1.slice(0, last);
    }
    S = S3;
    //--------------------------------------------------------------
    // STEPS 3,5: Select any rectangle j in S
    //--------------------------------------------------------------
    for (let por = 0; por < S.length; por++) {
      let j = S[por];
      let maxL = getMaxValue(L[j]);
      let I = new Uint32Array(L[j].length);
      last = 0;
      for (let i = 0; i < L[j].length; i++) {
        if (Math.abs(L[j][i] - maxL) < tolle) I[last++] = i;
      }
      let delta = (2 * maxL) / 3;
      let w = new Array(last);
      for (let r = 0; r < last; r++) {
        let i = I[r];
        let cm1 = C[j].slice();
        let cm2 = C[j].slice();
        cm1[i] += delta;
        cm2[i] -= delta;
        let xm1 = new Float32Array(cm1.length);
        let xm2 = new Float32Array(cm1.length);
        for (let i = 0; i < cm1.length; i++) {
          xm1[i] = xL[i] + cm1[i] * diffBorders[i];
          xm2[i] = xL[i] + cm2[i] * diffBorders[i];
        }
        let fm1 = fun(xm1);
        let fm2 = fun(xm2);
        funCalls += 2;
        w[r] = [Math.min(fm1, fm2), r];
        C.push(cm1, cm2);
        F.push(fm1, fm2);
      }
      let b = w.sort((a, b) => a[0] - b[0]);
      for (let r = 0; r < last; r++) {
        let u = I[b[r][1]];
        let ix1 = m + 2 * (b[r][1] + 1) - 1;
        let ix2 = m + 2 * (b[r][1] + 1);
        L[j][u] = delta / 2;
        L[ix1] = L[j].slice();
        L[ix2] = L[j].slice();
        let sumSquare = 0;
        for (let i = 0; i < L[j].length; i++) {
          sumSquare += Math.pow(L[j][i], 2);
        }
        D[j] = Math.sqrt(sumSquare);
        D[ix1] = D[j];
        D[ix2] = D[j];
      }
      m += 2 * last;
    }
    //--------------------------------------------------------------
    //                  Update
    //--------------------------------------------------------------
    // console.log('F', F)
    fMin = getMinValue(F);
    E =
      options.epsilon * Math.abs(fMin) > 1e-8
        ? options.epsilon * Math.abs(fMin)
        : 1e-8;

    iMin = getIndexOfMin(F, D, E, fMin);

    d = Array.from(new Set(D));
    d = d.sort((a, b) => a - b);

    dMin = new Array(d.length);
    for (let i = 0; i < d.length; i++) {
      let minIndex;
      let minValue = Number.MAX_SAFE_INTEGER;
      for (let k = 0; k < D.length; k++) {
        if (D[k] === d[i]) {
          if (F[k] < minValue) {
            minValue = F[k];
            minIndex = k;
          }
        }
      }
      dMin[i] = F[minIndex];
    }
    let currentMin = [];
    for (let j = 0; j < F.length; j++) {
      if (F[j] === fMin) {
        let temp = [];
        for (let i = 0; i < xL.length; i++) {
          temp[i] = xL[i] + C[j][i] * diffBorders[i];
        }
        currentMin.push(temp.slice());
      }
    }
    fs.appendFileSync('optimum', JSON.stringify(currentMin) + ',')
    t += 1;
  }
  //--------------------------------------------------------------
  //                  Saving results
  //--------------------------------------------------------------

  let result = {};
  result.minFunctionValue = fMin; // Best function value
  result.iterations = t; // Number of iterations

  for (let j = 0; j < m; j++) {
    // Transform to original coordinates
    for (let i = 0; i < xL.length; i++) {
      C[j][i] = xL[i] + C[j][i] * diffBorders[i];
    }
  }

  result.finalState = {C, F, D, L, d, dMin, funCalls};
  let xK = [];
  for (let i = 0; i < F.length; i++) {
    if (F[i] === fMin) {
      xK.push(C[i]);
    }
  }
  result.optimum = xK;
  return result;
}

module.exports = Direct;

function getIndexOfMin(F, D, E, fMin) {
  let index;
  let prevValue = Number.MAX_SAFE_INTEGER;
  for (let i = 0; i < F.length; i++) {
    let newValue = (F[i] - (fMin + E)) / D[i];
    if (newValue < prevValue) {
      index = i;
      prevValue = newValue;
    }
  }
  return index;
}

function getMinValue(F) {
  let nbPoints = F.length;
  let minValue = F[0];
  for (let i = 1; i < nbPoints; i++) {
    if (minValue > F[i])  minValue = F[i];
  }
  return minValue;
}

function getMaxValue(F) {
  let nbPoints = F.length;
  let maxValue = F[0];
  for (let i = 1; i < nbPoints; i++) {
    if (maxValue < F[i])  maxValue = F[i];
  }
  return maxValue;
}

// //--------------------------------------------------------
// //   Testing the algorithm with benchmark functions
// //-----------------------------------------------------

// function testFunction(x) {
//   let a =
//     x[1] -
//     (5 * Math.pow(x[0], 2)) / (4 * Math.pow(Math.PI, 2)) +
//     (5 * x[0]) / Math.PI -
//     6;
//   let b = 10 * (1 - 1 / (8 * Math.PI)) * Math.cos(x[0]) + 10;
//   let result = Math.pow(a, 2) + b;
//   return result;
// }

// let xL = [-5, 0];
// let xU = [10, 15];
// let options = { iterations: 6 };
// let result = Direct(testFunction, xL, xU, options);
// console.log('__________--------____________\n\n\n\n')
// console.log(result.optimum, result.iterations, result.finalState.C.length, result.finalState.funCalls)
// console.log('__________--------____________')
// console.log(result);
// let result2 = Direct(testFunction, xL, xU, options, result.finalState);
// console.log(result2.optimum, result2.iterations, result2.finalState.C.length, result2.finalState.funCalls)
// for (let i = 0; i < 19; i++) {
//   console.time('hola');
//   let result = Direct(testFunction, xL, xU, GLOBAL);
//   console.timeEnd('hola');
// }
// console.log('-___----------_____RESULT-____----------___');
// console.log(rCesult);
