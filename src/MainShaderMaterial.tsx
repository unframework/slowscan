import { extend, ReactThreeFiber } from '@react-three/fiber';
import { shaderMaterial } from '@react-three/drei';

import vert from './MainShaderMaterial.vert.glsl';
import frag from './MainShaderMaterial.frag.glsl';

export const MainShaderMaterial = shaderMaterial({}, vert, frag);

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
