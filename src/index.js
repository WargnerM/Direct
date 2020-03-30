'use strict';

let Matrix = require('ml-matrix').Matrix;
let conhull = require('../src/conhull.js');

/**
 * Creates new PCA (Principal Component Analysis) from the dataset
 * @param {Matrix} fun - dataset or covariance matrix.
 * @param {Array} xU - dataset or covariance matrix.
 * @param {Array} xL - dataset or covariance matrix.
 * @param {Object} [options]
 * @param {number} [options.printLevel] - Parameter to show information.
 * @param {Object} [options.global] - Parameter with information.
 * */


function Direct(fun, xL, xU, options = {}, entries = {}) {
  const opts = {
    iterations: 50,
    epsilon: 1e-4,
    tol: 0.01,
  };

  let global = entries;
  options = Object.assign({}, opts, options);

  if (fun === undefined || xL === undefined || xU === undefined) {
    throw new Error('There is something undefined');
  }

  if (xL.length !== xU.length) {
    throw new Error('Lower bounds and Upper bounds for x are not of the same length');
  }
  if (global.C) {
    global.C = Matrix.checkMatrix(global.C);
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
  let tolle2 = 1e-12;
  let dMin = global.dMin;
 
  let F, m, D, L, d, fMin, E, minIndex, iMin, f0, C;
  if (global.C && global.C.length !== 0) {
    F = global.F;
    m = F.length;
    D = global.D;
    L = global.L;
    d = global.d;
    dMin = global.dMin;
    epsilon = options.epsilon;
    fMin = Math.min(...F);
    E = options.epsilon * Math.abs(fMin) > 1e-8 ? options.epsilon * Math.abs(fMin) : 1e-8;
    let test = F.map((x) => x - (fMin + E));
    let difference = [];
    for (let i = 0; i < test.length; i++) {
      difference[i] = test[i] - D[i];
    }
    minIndex = Math.min(...difference);
    for (let i = 0; i < m; i++) {
      C[i] = global.C(i).map((x) => x - (xL / (xU - xL)));
    }
  } else {
    m = 0;
    C = [new Array(n).fill(0.5)];
    let xM = [];
    for (let i = 0; i < xL.length; i++) {
      xM[i] = xL[i] + C[0][i] * (xU[i] - xL[i]);
    }
    fMin = fun(xM);
    funCalls = funCalls + 1;
    iMin = 1;
    L = [new Array(n).fill(0.5)];
    D = 0;
    for (let i = 0; i < L.length; i++) {
      D += Math.pow(L[0][i], 2)
    }
    D = [Math.sqrt(D)];
    F = [fMin];
    d = D;
    dMin = fMin;
  }
  let S = [];
  let S_1 = [];
  let S_2 = [];
  let S_3 = [];
  let t = 0;

  //-------------------------------------------------------------------------
  //                          Iteration loop
  //-------------------------------------------------------------------------
  while (t < options.iterations) {
    //----------------------------------------------------------------------
    //  STEP 2. Identify the set S of all potentially optimal rectangles
    //----------------------------------------------------------------------
    let idx;
    for (let i = 0; i < D.length; i++) {
      if (D[i] === d[i]) idx = i;
    }
    for (let i = idx; i < d.length; i++) {
      if (typeof D.findIndex(x => x == d[i]) == 'number' && typeof F.findIndex(x => x == dMin) == 'number') S_1.push(i);
    }
    if (d.length - idx > 1) {
      let a1 = D[iMin];
      let b1 = F[iMin];
      let a2 = d[d.length];
      let b2 = dMin[d.length];
      let slope = (b2 - b1) / (a2 - a1);
      let constant = b1 - slope * a1;
      for (let i = 0; S_1.length; i++) {
        let j = S_1[i];
        if (F[j] <= slope * D[j] + constant + tolle2) {
          S_2.push(j);
        }
      }
      let xx, yy;
      for (let i = 0; i < S_2.length; i++) {
        xx = D[S_2[i]];
        yy = F[S_2[i]];
      }
      let h = conhull(xx, yy);
      for (let i = 0; i < h.length; i++) {
        S_3 = S_2[h[i]];
      }
    } else {
      S_3 = S_1;
    }
    S = S_3;
    
    //--------------------------------------------------------------
    // STEPS 3,5: Select any rectangle j in S
    //--------------------------------------------------------------
    for (let por = 0; por < S.length; por++) {
      let j = S[por];
      let maxL = Math.max(...L[j]);
      let I = [];
      for (let i = 0; i < L[j].length; i++) {
        if (Math.abs(L[j][i] - maxL) < tolle) I.push(i)
      }
      let delta = (2 * maxL)/3;
      let w = [];
      for (let r = 0; r < I.length; r++) {
        let i = I[r] + 1;
        let eI = [1, 1];
        if (i-1 > 0) {
          eI[0] = 0;
        } if (n - i > 0) {
          eI[1] = 0;
        }
        let cm1 = []; let cm2 = [];
        for (let i = 0; i < C[j].length; i++) {
          cm1[i] = C[j][i] + (delta * eI[i]);
          cm2[i] = C[j][i] - (delta * eI[i]);
        }
        let xm1 = []; let xm2 = [];
        for (let i = 0; i < cm1.length; i++) {
          xm1[i] = xL[i] + (cm1[i] * (xU[i] - xL[i]));
          xm2[i] = xL[i] + (cm2[i] * (xU[i] - xL[i]));
        }
        let fm1 = fun(xm1);
        let fm2 = fun(xm2);
        funCalls++*2;
        console.log(funCalls)
        w[r] = Math.min(fm1, fm2);
        C.push(cm1, cm2);
        F.push(fm1, fm2);
      }

    let ws = w.slice().sort((a, c) => a - c)
    let b = w.map((a, c) => ws.findIndex(y => y == w[c]));

      for (let r = 0; r < I.length; r++) {
        let u = I[b[r]];
        let ix1 = m + (2 * (b[r] + 1)) - 1;
        let ix2 = m + (2 * (b[r] + 1));
        L[j][u] = delta / 2;
        L[ix1] = L[j].slice()
        L[ix2] = L[j].slice()
        let item = 0
        for (let i = 0; i < L[j].length; i++) {
          item += L[j][i] * L[j][i]
        }
        D[j] = Math.sqrt(item);
        D[ix1] = D[j];
        D[ix2] = D[j];
      }
      console.log(D)
      m = m + (2 * I.length);
    }
    //--------------------------------------------------------------
    //                  Update
    //--------------------------------------------------------------
    
    fMin = Math.min(...F);
    E = options.epsilon * Math.abs(fMin) > 1e-8 ? options.epsilon * Math.abs(fMin) : 1e-8;
    let item = [];
    for (let i = 0; i < F.length; i++) {
      item[i] = (F[i] - (fMin + E))/D[i]
    }
  
    let minIndex = item.findIndex(x => x == Math.min(...item));
    let counter = 0;
    let st = 1;
    while (counter < st) {
      let dTmp = d[counter];
      let idx = d.filter(x => x !== dTmp);
      d = [dTmp, ...idx];
      st = d.length;
      counter++
    }
    
    d = d.sort((a, b) => a - b);
    dMin = [];
    for (let i = 0; i < d.length; i++) {
      let idx1 = [];
      D.forEach(function(a, b) {
        if (a === d[i]) {
          idx1.push(b)
        }
      }, idx1)
      let fTmp = [];
      for (let j = 0; j < idx1.length; j++) {
        fTmp[j] = F[idx1[j]];
        dMin[i] = Math.min(...fTmp);
      }
    }
    console.log(dMin)
    t++
  }

  //--------------------------------------------------------------
  //                  Saving results
  //--------------------------------------------------------------
  
  let result = {};
  result.f_k = fMin; // Best function value
  result.Iter = iterations; // Number of iterations

  CC = new Matrix(C.rows, C.columns);
  for (let i = 0; i < m; i++) { // Transform to original coordinates
    let diff = x_U.sub(x_L);
    let subMatrix = C.getColumnVector(i);
    let newColumn = x_L.addM(subMatrix.mulM(diff)); // x_L + C(:,i).*(x_U-x_L)
    CC.setColumn(i, newColumn);
  }
  result.GLOBAL = {};
  result.GLOBAL.C = CC; // All sampled points in original coordinates
  result.GLOBAL.F = F; // All function values computed
  result.GLOBAL.D = D; // All distances
  result.GLOBAL.L = L; // All lengths
  result.GLOBAL.d = d;
  result.GLOBAL.d_min = d_min;
  result.FuncEv = nFunc;
  // Find all points i with F(i)=f_min

  let index = 0;
  let xK = new Matrix();
  for (let i = 0; i < F.columns; i++) {
    if (F[i] === f_min) xK.addColumn(index++, C.getColumn(i));
  }
  result.x_k = xK;
}

//--------------------------------------------------------
//   Testing the algorithm with benchmark functions
//-----------------------------------------------------

function testFunction(x) {
  let a = (x[1] - 5 * Math.pow(x[0], 2) / (4 * Math.pow(Math.PI, 2)) + 5 * x[0] / Math.PI - 6);
  let b = (10 * (1 - 1 / (8 * Math.PI)) * Math.cos(x[0]) + 10);
  let result = Math.pow(a, 2) + b;
  return result;
}

let xL = [-5, 0];
let xU = [10, 15];
let GLOBAL = { iterations: 1 };
let result = Direct(testFunction, xL, xU, GLOBAL);
