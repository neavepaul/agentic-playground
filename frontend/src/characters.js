import * as THREE from 'three';

const outfits = {
  moira: ['#d879a0', '#643f36'],
  neave: ['#7da5de', '#403332'],
  minu: ['#82baa0', '#302b36'],
  paul: ['#d9ac62', '#797078'],
};

export function createCharacter(id) {
  const group = new THREE.Group();
  const [shirt, hair] = outfits[id] ?? ['#aa94cc', '#514139'];
  const materials = {};
  function material(color) {
    return materials[color] ??= new THREE.MeshStandardMaterial({ color, roughness: .8 });
  }
  function puff(parent, color, position, scale) {
    const mesh = new THREE.Mesh(new THREE.SphereGeometry(1, 20, 16), material(color));
    mesh.position.set(...position);
    mesh.scale.set(...scale);
    mesh.castShadow = true;
    parent.add(mesh);
    return mesh;
  }
  const skin = '#f2c5a5';
  for (const x of [-.105, .105]) {
    puff(group, '#455366', [x, .13, .035], [.095, .115, .14]);
  }
  puff(group, shirt, [0, .47, 0], [.22, .29, .18]);
  puff(group, '#fff2df', [0, .62, .162], [.075, .035, .025]);
  for (const y of [.43, .53]) puff(group, '#fff2df', [0, y, .18], [.018, .018, .012]);

  const head = new THREE.Group();
  head.position.y = .96;
  group.add(head);
  puff(head, skin, [0, 0, 0], [.27, .27, .235]);
  for (const x of [-.26, .26]) puff(head, skin, [x, -.01, 0], [.045, .065, .05]);
  const cap = new THREE.Mesh(new THREE.SphereGeometry(.278, 24, 16, 0, Math.PI * 2, 0, Math.PI * .49), material(hair));
  cap.scale.z = .9;
  cap.position.y = .015;
  cap.castShadow = true;
  head.add(cap);
  for (const [x, y] of [[-.15, .12], [-.04, .16], [.08, .17]]) {
    puff(head, hair, [x, y, .16], [.09, .075, .07]);
  }
  for (const x of [-.09, .09]) {
    puff(head, '#34313d', [x, .005, .22], [.032, .043, .022]);
    puff(head, '#ffffff', [x - .009, .02, .24], [.009, .012, .006]);
    puff(head, '#e89999', [x * 1.6, -.06, .195], [.044, .024, .016]);
  }
  puff(head, skin, [0, -.045, .236], [.032, .033, .03]);
  const smile = new THREE.Mesh(new THREE.TorusGeometry(.06, .009, 6, 16, Math.PI), material('#92594f'));
  smile.rotation.z = Math.PI;
  smile.position.set(0, -.075, .224);
  head.add(smile);

  // A raised arm with a shoulder pivot gives a readable little greeting.
  const arm = new THREE.Group();
  arm.position.set(-.18, .61, 0);
  group.add(arm);
  puff(arm, shirt, [0, .075, 0], [.065, .11, .07]);
  puff(arm, skin, [0, .21, 0], [.052, .085, .05]);
  puff(arm, skin, [0, .295, 0], [.065, .07, .04]);
  puff(group, shirt, [.22, .49, 0], [.065, .13, .07]);
  puff(group, skin, [.24, .35, .015], [.055, .065, .05]);
  const phase = [...id].reduce((value, char) => value + char.charCodeAt(0), 0) % 17;
  group.userData.animate = (time, camera, reducedMotion) => {
    const dx = camera.position.x - group.position.x;
    const dz = camera.position.z - group.position.z;
    group.rotation.y = Math.atan2(dx, dz);
    head.rotation.x = -Math.atan2(camera.position.y - .96, Math.hypot(dx, dz)) * .65;
    const seconds = time / 1000 + phase;
    const greeting = seconds % 6 < 2.5;
    arm.rotation.z = .3 + (!reducedMotion && greeting ? Math.sin(seconds * 9) * .22 : 0);
    head.rotation.z = reducedMotion ? 0 : Math.sin(seconds * 1.6) * .035;
  };
  return group;
}
