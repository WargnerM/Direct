
/**
 * Creates new PCA (Principal Component Analysis) from the dataset
 * @param {Matrix} fun - dataset or covariance matrix.
 * @param {Array} xU - dataset or covariance matrix.
 * @param {Array} xL - dataset or covariance matrix.
 * @param {Object} [options]
 * @param {number} [options.printLevel] - Parameter to show information.
 * @param {Object} [options.global] - Parameter with information.
 * */


export function gblSolve(fun, xL, xU, options = {}, entries = {}) {
  const opts = {
    iterations: 50,
    epsilon: 1e-4,
    tol: 0.01,
  };

  let global = entries;
  options = Object.assign({}, options, opts);

  if (fun === undefined || xL === undefined || xU === undefined) {
    throw new Error('There is something undefined');
  }

  if (xL.length !== xU.length) {
    throw new Error('Lower bounds and Upper bounds for x are not of the same length');
  }

  // *** Step 1: Inicialization
  let funCalls = true;
  let convFlag = 0;
  let n = xL.length;
  let tolle = 1e-16;
  let tolle2 = 1e-12;
  let dMin = global.dMin;

  let F, m, D, L, d, fMin, epsilon, E, minIndex, iMin, f0;
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
    let C = [];
    for (let i = 0; i < m; i++) {
      C[i] = global.C[i].map((x) => x - (xL[i] / (xU[i] - xL[i])));
    }
  } else {
    m = 1;
    let C = new Array(n).fill(0.5);
    let xM = [];
    for (let i = 0; i < xL.length; i++) {
      xM[i] = xL[i] + C[i] * (xU[i] - xL[i]);
    }
    let fMin = fun(xM);
    f0 = fMin;
    funCalls = funCalls + 1;
    iMin = 1;
    let L = new Array(n).fill(0.5);
    let D = Math.sqrt(
      L.reduce((a, b) => a + Math.pow(b, 2))
    );
    F = [fMin];
    d = D;
    dMin = fMin;
  }

  let S = [];
  let S_1 = [];
  let S_2 = [];
  let S_3 = [];
  let t = 1;
  while (t <= options.iterations) {
    let idx;
    for (let i = 0; i < D.length; i++) {
      if (D[iMin] === d[i]) idx = i;
    }
    let idx2;
    for (let i = idx; i < d.length; i++) {
      if (D === d[i] && F === dMin[i]) idx2 = i;
      S_1[i] = idx2;
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
    t = t + 1;
  }

  // saving results

  let result = {};
  result.f_k = f_min; // Best function value
  result.Iter = T; // Number of iterations

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

