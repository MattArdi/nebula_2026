// Fleet-wide rankings shown by the top navigation buttons — up to 10 trains
// per subsystem, all hardcoded (none of it comes from a model run) and shown
// in ranked order, worst first.
export const FLEET_RANKINGS = {
  door: {
    valueHeader: "Predicted no. of abnormalities",
    rows: [
      { id: "6767", value: "30 (10 Close | 20 Open)" },
      { id: "6969", value: "24 (14 Close | 10 Open)" },
      { id: "6868", value: "20 (10 Close | 10 Open)" },
      { id: "7171", value: "17 (9 Close | 8 Open)" },
      { id: "8383", value: "15 (8 Close | 7 Open)" },
      { id: "7676", value: "12 (5 Close | 7 Open)" },
      { id: "9595", value: "10 (6 Close | 4 Open)" },
      { id: "7373", value: "8 (3 Close | 5 Open)" },
      { id: "9696", value: "5 (2 Close | 3 Open)" },
      { id: "8686", value: "3 (1 Close | 2 Open)" },
    ],
  },
  acv: {
    valueHeader: "Faulty Car",
    rows: [
      { id: "6969", value: "Car 01" },
      { id: "6767", value: "Car 06" },
      { id: "6868", value: "Car 07" },
      { id: "7171", value: "Car 03" },
      { id: "8383", value: "Car 05" },
      { id: "7676", value: "Car 02" },
      { id: "9595", value: "Car 08" },
      { id: "7373", value: "Car 04" },
      { id: "9696", value: "Car 06" },
      { id: "8686", value: "Car 01" },
    ],
  },
  shm: {
    valueHeader: "Current Train Damage",
    // The loaded train (8686) is first; its value here is a placeholder, the
    // live one is shown in its place.
    rows: [
      { id: "8686", value: "0.103936706985" },
      { id: "7676", value: "0.089234798" },
      { id: "9696", value: "0.061735412" },
      { id: "6767", value: "0.055120873" },
      { id: "9595", value: "0.047389216" },
      { id: "7171", value: "0.041204536" },
      { id: "6969", value: "0.039817452" },
      { id: "8383", value: "0.033562904" },
      { id: "7373", value: "0.028117645" },
      { id: "6868", value: "0.024408167" },
    ],
  },
  rail: {
    valueHeader: "No. of Abnormalities",
    rows: [
      { id: "7676", value: "21 (9 Side I | 12 Side II)" },
      { id: "8686", value: "17 (7 Side I | 10 Side II)" },
      { id: "7171", value: "13 (6 Side I | 7 Side II)" },
      { id: "6969", value: "11 (4 Side I | 7 Side II)" },
      { id: "6767", value: "9 (2 Side I | 7 Side II)" },
      { id: "8383", value: "8 (3 Side I | 5 Side II)" },
      { id: "9696", value: "7 (1 Side I | 6 Side II)" },
      { id: "9595", value: "5 (2 Side I | 3 Side II)" },
      { id: "7373", value: "3 (1 Side I | 2 Side II)" },
      { id: "6868", value: "1 (0 Side I | 1 Side II)" },
    ],
  },
};
