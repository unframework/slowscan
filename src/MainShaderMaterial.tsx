import { extend, ReactThreeFiber } from '@react-three/fiber';
import { shaderMaterial } from '@react-three/drei';

import vert from './MainShaderMaterial.vert.glsl';
import frag from './MainShaderMaterial.frag.glsl';

export const MainShaderMaterial = shaderMaterial(
  {
    mouse: [0, 0],
    resolution: [100, 100],
    blueNoise: null
  },
  vert,
  frag
);

extend({ MainShaderMaterial });

declare global {
  namespace JSX {
    interface IntrinsicElements {
      mainShaderMaterial: ReactThreeFiber.Object3DNode<
        typeof MainShaderMaterial,
        {}
      >;
    }
  }
}
