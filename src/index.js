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
  let diffBorders = xU.map((x, i) => x - xL[i]);
 
  let F, m, D, L, d, fMin, E, iMin, minIndex, f0, C;
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
    let item = [];
    for (let i = 0; i < F.length; i++) {
      item[i] = (F[i] - (fMin + E))/D[i]
    }
    iMin = item.findIndex(x => x == Math.min(...item));
    for (let i = 0; i < m; i++) {
      C[i] = global.C(i).map((x) => x - (xL / (xU - xL)));
    }
  } else {
    m = 0;
    C = [new Array(n).fill(0.5)];
    let xM = [];
    for (let i = 0; i < xL.length; i++) {
      xM[i] = xL[i] + C[0][i] * diffBorders[i];
    }
    fMin = fun(xM);
    funCalls = funCalls + 1;
    iMin = 0;
    L = [ new Array(n).fill(0.5) ];
    D = [ Math.sqrt(n * Math.pow(0.5, 2)) ];
    F = [ fMin ];
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
    let S = [];
    let S1 = [];
    let S2 = [];
    let S3 = [];
    let idx = d.find((e) => e === D[iMin]);
    let a = 0;

    for (let i = idx; i < d.length; i++) {
      let idx2 = [];
      for (let f = 0; f < F.length; f++) {
        if (F[f] === dMin[i]) {
          if (D[f] === d[i]) idx2.push(f);
        }
      }
      S1 = S1.concat(idx2);
    }

    if (d.length - idx > 1) { // toTest the condition
      let a1 = D[iMin];
      let b1 = F[iMin];
      let a2 = d[d.length - 1];
      let b2 = dMin[d.length - 1];
      let slope = (b2 - b1) / (a2 - a1);
      let constant = b1 - slope * a1;
      for (let i = 0; i < S1.length; i++) {
        let j = S1[i];
        if (F[j] <= slope * D[j] + constant + tolle2) {
          S2.push(j);
        }
      }
      let xx, yy;
      for (let i = 0; i < S2.length; i++) {
        xx = D[S2[i]];
        yy = F[S2[i]];
      }
      let h = conhull(xx, yy);
      for (let i = 0; i < h.length; i++) {
        S3 = S2[h[i]];
      }
    } else {
      S3 = S1;
    }
    S = S3;
    
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
        let i = I[r]; //I[r] + 1; why plus one? i is the index of the dimension that will be splitted
        let cm1 = C[j].slice();
        let cm2 = C[j].slice();
        cm1[i] += delta;
        cm2[i] -= delta;

        let xm1 = []; let xm2 = [];
        for (let i = 0; i < cm1.length; i++) {
          xm1[i] = xL[i] + (cm1[i] * diffBorders[i]);
          xm2[i] = xL[i] + (cm2[i] * diffBorders[i]);
        }
        let fm1 = fun(xm1);
        let fm2 = fun(xm2);
        funCalls += 2;
        w[r] = [Math.min(fm1, fm2), r];
        C.push(cm1, cm2);
        F.push(fm1, fm2);
      }

      let b = w.sort((a, b) => a[0] - b[0]);

      for (let r = 0; r < I.length; r++) {
        let u = I[b[r][1]];
        let ix1 = m + (2 * (b[r] + 1)) - 1;
        let ix2 = m + (2 * (b[r] + 1));
        L[j][u] = delta / 2;
        L[ix1] = L[j].slice();
        L[ix2] = L[j].slice();
        let sumSquare = 0
        for (let i = 0; i < L[j].length; i++) {
          sumSquare += Math.pow(L[j][i], 2);
        }
        D[j] = Math.sqrt(sumSquare);
        D[ix1] = D[j];
        D[ix2] = D[j];
      }
      m += 2 * I.length;
    }
    //--------------------------------------------------------------
    //                  Update
    //--------------------------------------------------------------
    fMin = Math.min(...F);
    E = options.epsilon * Math.abs(fMin) > 1e-8 ? options.epsilon * Math.abs(fMin) : 1e-8;
    
    let prevValue = Number.MAX_SAFE_INTEGER;
    for (let i = 0; i < F.length; i++) {
      let newValue = (F[i] - (fMin + E))/D[i];
      if (newValue < prevValue) {
        iMin = i;
        prevValue = newValue;
      }
    }
    

    for (let i = 0; i !== d.length; i++) {
      let dTmp = d[i];
      let idx = [];
      for (let di = 0; di < d.length; di++) {
        if (d[di] !== dTmp) idx.push(di);
      }
      let newD = new Array(idx.length + 1);
      newD[0] = dTmp;
      for (let k = 0, kk = 1; k < idx.length; k++) {
        newD[kk++] = d[idx[k]];
      }
      d = newD;
    }
    
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
    t++
  }
  
  //--------------------------------------------------------------
  //                  Saving results
  //--------------------------------------------------------------
  /*
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
  */
  
  /*
  result.global = {};
  global.C = C; // All sampled points in original coordinates
  global.F = F; // All function values computed
  global.D = D; // All distances
  global.L = L; // All lengths
  global.d = d;
  global.dMin = dMin;
  global.funCalls = funCalls;
  // Find all points i with F(i)=f_min

  let index = 0;
  let xK = [];
  for (let i = 0; i < F.columns; i++) {
    if (F[i] === f_min) xK[index] = C[i]
  }
  result.x_k = xK;
  */
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
let GLOBAL = { iterations: 2 };
let result = Direct(testFunction, xL, xU, GLOBAL