function parseData(xy, threshold) {
  var nbSeries = xy.length;
  var t = null;
  var y_data = null;
  var x;
  var y;
  var maxY = 0;
  var i;
  var j;

  if (nbSeries == 2) {
    var nbPoints = xy[0].length;
    t = new Array(nbPoints);
    y_data = new Array(nbPoints);
    x = xy[0];
    y = xy[1];
    if (typeof x[0] === 'number') {
      for (i = 0; i < nbPoints; i++) {
        t[i] = x[i];
        y_data[i] = y[i];
        if (y[i] > maxY) maxY = y[i];
      }
    } else {
      if (typeof x[0] === 'object') {
        for (i = 0; i < nbPoints; i++) {
          t[i] = x[i];
          y_data[i] = y[i];
          if (y[i] > maxY) maxY = y[i];
        }
      }
    }
  } else {
    var nbPoints = nbSeries;
    t = new Array(nbPoints);
    y_data = new Array(nbPoints);
    for (i = 0; i < nbPoints; i++) {
      t[i] = xy[i];
      y_data[i] = xy[i];
      if (y_data[i] > maxY) maxY = y_data[i];
    }
  }
  for (i = 0; i < nbPoints; i++) {
    y_data[i] /= maxY;
  }
  if (threshold) {
    for (i = nbPoints - 1; i >= 0; i--) {
      if (y_data[i] < threshold) {
        y_data.splice(i, 1);
        t.splice(i, 1);
      }
    }
  }
  if (t.length > 0) {
    return [
      t,
      y_data,
      maxY
    ];
  }
  return null;
}

module.exports = parseData
