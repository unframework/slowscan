uniform vec2 mouse;
uniform vec2 resolution;
uniform sampler2D blueNoise;

const int MAX_MARCHING_STEPS = 100;
const float MIN_DIST = 0.0;
const float MAX_DIST = 10.0;
const float EPSILON = 0.0001;

#define VOL_AMOUNT 0.25
#define VOL_STEP_D 0.5

#define PI 3.14159265359

/**
 * Signed distance function for a cube centered at the origin
 * with width = height = length = 2.0
 */
float cubeSDF(vec3 p, vec3 center, vec3 dim) {
    // If d.x < 0, then -1 < p.x < 1, and same logic applies to p.y, p.z
    // So if all components of d are negative, then p is inside the unit cube
    vec3 d = abs(p - center) - dim;

    // Assuming p is inside the cube, how far is it from the surface?
    // Result will be negative or zero.
    float insideDistance = min(max(d.x, max(d.y, d.z)), 0.0);

    // Assuming p is outside the cube, how far is it from the surface?
    // Result will be positive or zero.
    float outsideDistance = length(max(d, 0.0));

    return insideDistance + outsideDistance;
}

/**
 * Signed distance function for a sphere centered at the origin with radius 1.0;
 */
float sphereSDF(vec3 p) {
    return length(p) - 1.0;
}

/**
 * Signed distance function describing the scene.
 *
 * Absolute value of the return value indicates the distance to the surface.
 * Sign indicates whether the point is inside or outside the surface,
 * negative indicating inside.
 */
float sceneSDF(vec3 samplePoint) {
    float cube1 = cubeSDF(samplePoint, vec3(0.0, -0.2, 0.0), vec3(4.0, 0.2, 4.0));
    float cube2 = cubeSDF(samplePoint, vec3(0.0, 0.75, -0.5), vec3(0.5, 0.75, 1.0));

    return min(cube1, cube2);
}

/**
 * Return the shortest distance from the eyepoint to the scene surface along
 * the marching direction. If no part of the surface is found between start and end,
 * return end.
 *
 * eye: the eye point, acting as the origin of the ray
 * marchingDirection: the normalized direction to march in
 * start: the starting distance away from the eye
 * end: the max distance away from the ey to march before giving up
 */
float shortestDistanceToSurface(vec3 eye, vec3 marchingDirection, float start, float end) {
    float depth = start;
    for (int i = 0; i < MAX_MARCHING_STEPS; i++) {
        float dist = sceneSDF(eye + depth * marchingDirection);
        if (dist < EPSILON) {
      return depth;
        }
        depth += dist;
        if (depth >= end) {
            return end;
        }
    }
    return end;
}


/**
 * Return the normalized direction to march in from the eye point for a single pixel.
 *
 * fieldOfView: vertical field of view in degrees
 * size: resolution of the output image
 * fragCoord: the x,y coordinate of the pixel in the output image
 */
vec3 rayDirection(float fieldOfView, vec2 size, vec2 fragCoord) {
    //Fragment coords remapped to -1,1 range
    vec2 screenPos = -1.0 + 2.0 * fragCoord.xy / size;
    //Aspect correction
  screenPos.x *= resolution.x / resolution.y;

    //Calculate ray direction for current fragment
    float aperture = 0.25 * 2.0*PI;
    float f = 1.0/aperture;
    float r = length(screenPos);
    float phi = atan(screenPos.y, screenPos.x);
    float theta = atan(r / (3.0 * f)) * 2.0;

    return (vec3(sin(theta) * cos(phi), sin(theta) * sin(phi), -cos(theta)));

    //vec2 xy = fragCoord - size / 2.0;
    //float z = size.y / tan(radians(fieldOfView) / 2.0);
    //return normalize(vec3(xy, -z));
}

/**
 * Using the gradient of the SDF, estimate the normal on the surface at point p.
 */
vec3 estimateNormal(vec3 p) {
    return normalize(vec3(
        sceneSDF(vec3(p.x + EPSILON, p.y, p.z)) - sceneSDF(vec3(p.x - EPSILON, p.y, p.z)),
        sceneSDF(vec3(p.x, p.y + EPSILON, p.z)) - sceneSDF(vec3(p.x, p.y - EPSILON, p.z)),
        sceneSDF(vec3(p.x, p.y, p.z  + EPSILON)) - sceneSDF(vec3(p.x, p.y, p.z - EPSILON))
    ));
}

/**
 * Lighting contribution of a single point light source via Phong illumination.
 *
 * The vec3 returned is the RGB color of the light's contribution.
 *
 * k_a: Ambient color
 * k_d: Diffuse color
 * k_s: Specular color
 * alpha: Shininess coefficient
 * p: position of point being lit
 * eye: the position of the camera
 * lightPos: the position of the light
 * lightIntensity: color/intensity of the light
 *
 * See https://en.wikipedia.org/wiki/Phong_reflection_model#Description
 */
vec3 contribForLight(vec3 p, vec3 N, vec3 lightPos, vec3 lightIntensity) {
    vec3 L = normalize(lightPos - p);

    float dotLN = dot(L, N);

    if (dotLN < 0.0) {
        // Light not visible from this point on the surface
        return vec3(0.0, 0.0, 0.0);
    }

    return lightIntensity * dotLN;
}

vec3 contribForSun(vec3 p, vec3 N, vec3 albedo) {
    vec3 dir = normalize(vec3(-1.0, 2.0, 1.0));

    float dotLN = dot(dir, N);

    if (dotLN < 0.0) {
        // Light not visible from this point on the surface
        return vec3(0.0, 0.0, 0.0);
    }

    // shadow mask check
    vec2 groundXY = p.xz - p.y * dir.xz / dir.y;
    if (abs(groundXY.x) > 1.0 || abs(groundXY.y) > 1.0) {
        // occluded by shadow mask
        return vec3(0.0, 0.0, 0.0);
    }

    // occlusion check (starting a bit off the surface)
    float surfaceDist = shortestDistanceToSurface(p, dir, 0.01, MAX_DIST);
    if (surfaceDist < MAX_DIST - EPSILON) {
        // occluded by hard surface
        return vec3(0.0, 0.0, 0.0);
    }

    // falloff from close-by "atmosphere edge"
    // (we modify transmittance to be closer to 1 just aesthetically)
    float distToEdge = max(0.0, 4.0 - p.z);
    float transmittance = 1.0 - (1.0 - exp(-VOL_AMOUNT * distToEdge)) * 0.2;

    return vec3(2.2, 2.2, 2.1) * albedo * transmittance * dotLN;
}

// from http://www.aduprat.com/portfolio/?page=articles/hemisphericalSDFAO
vec3 randomSphereDir(vec4 rand) {
    float s = rand.x * PI * 2.0;
    float t = rand.y * 2.0 - 1.0;
    return vec3(sin(s), cos(s), t) / sqrt(1.0 + t * t);
}

vec3 randomHemisphereDir(vec3 dir, vec4 rand) {
    vec3 v = randomSphereDir(rand);
    return v * sign(dot(v, dir));
}

#define BOUNCE_COUNT 2
#define BOUNCE_SAMPLE_COUNT 4

vec3 contribForBounce(vec3 p, vec3 N, vec3 albedo, vec4 rand) {
    vec3 bounceTotal = vec3(0.0, 0.0, 0.0);
    float sampleFraction = 1.0 / float(BOUNCE_SAMPLE_COUNT);
    vec3 comboAlbedo = albedo;

    for (int bsi = 0; bsi < BOUNCE_SAMPLE_COUNT; bsi += 1) {
        vec3 currentP = p;
        vec3 currentN = N;

        for (int bi = 0; bi < BOUNCE_COUNT; bi += 1) {
            vec4 sampleRand = texture(blueNoise, rand.xy + float(bsi) * sampleFraction + float(bi) * 0.1);

            // pick random direction for bounce origin
            vec3 bounceDir = randomHemisphereDir(currentN, sampleRand);

            // follow to any hard surface that could have produced the bounce
            // @todo use shorter cutoff
            float surfaceDist = shortestDistanceToSurface(currentP, bounceDir, 0.01, MAX_DIST);
            if (surfaceDist > MAX_DIST - EPSILON) {
                break;
            }

            // get bounce origin surface lighting
            // @todo apply proper albedo
            vec3 bounceP = currentP + surfaceDist * bounceDir;
            vec3 bounceN = estimateNormal(bounceP);
            bounceTotal += contribForSun(bounceP, bounceN, comboAlbedo) * sampleFraction;

            // repeat another bounce
            currentP = bounceP;
            currentN = bounceN;
            comboAlbedo *= albedo; // combine two albedos
        }
    }

    return bounceTotal * 1.0; // amplify for visual effect
}

/**
 * Lighting via Phong illumination.
 *
 * The vec3 returned is the RGB color of that point after lighting is applied.
 * k_a: Ambient color
 * k_d: Diffuse color
 * k_s: Specular color
 * alpha: Shininess coefficient
 * p: position of point being lit
 * eye: the position of the camera
 *
 * See https://en.wikipedia.org/wiki/Phong_reflection_model#Description
 */
vec3 illumination(vec3 p, vec3 eye, vec3 albedo, vec4 rand) {
    vec3 N = estimateNormal(p);

    vec3 color = vec3(0.1, 0.1, 0.1);

    /*
    vec3 light1Pos = vec3(4.0 + sin(iTime),
                          2.0,
                          4.0 + cos(iTime));
    vec3 light1Intensity = vec3(0.4, 0.4, 0.4);

    color += contribForLight(p, N,
                                  light1Pos,
                                  light1Intensity);

    vec3 light2Pos = vec3(2.0 + sin(0.37 * iTime),
                          2.0 + cos(0.37 * iTime),
                          2.0);
    vec3 light2Intensity = vec3(0.4, 0.4, 0.4);

    color += contribForLight(p, N,
                                  light2Pos,
                                  light2Intensity);
    */

    color += contribForSun(p, N, albedo);
    color += contribForBounce(p, N, albedo, rand);

    return color;
}

/**
 * Return a transform matrix that will transform a ray from view space
 * to world coordinates, given the eye point, the camera target, and an up vector.
 *
 * This assumes that the center of the camera is aligned with the negative z axis in
 * view space when calculating the ray marching direction. See rayDirection.
 */
mat4 getViewMatrix(vec3 eye, vec3 center, vec3 up) {
    // Based on gluLookAt man page
    vec3 f = normalize(center - eye);
    vec3 s = normalize(cross(f, up));
    vec3 u = cross(s, f);
    return mat4(
        vec4(s, 0.0),
        vec4(u, 0.0),
        vec4(-f, 0.0),
        vec4(0.0, 0.0, 0.0, 1)
    );
}

void main( )
{
    vec4 blueRand = texture(blueNoise, gl_FragCoord.xy / vec2(1024.0));

  vec3 viewDir = rayDirection(60.0, resolution, gl_FragCoord.xy);

    vec2 mouse = 2.0 * (mouse / resolution - 0.5);
    float eyeAngleX = 6.28 * mouse.x;
    float eyeAngleY = mouse.y * 6.28;
    vec2 eyeXY = vec2(-sin(eyeAngleX), cos(eyeAngleX));
    vec3 eye = vec3(eyeXY * cos(eyeAngleY), sin(eyeAngleY)).xzy * 3.0;
    //vec3 eye = vec3((mouse / resolution - 0.5) * 4.0, 2.0);

    mat4 viewToWorld = getViewMatrix(eye, vec3(0.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0));

    vec3 worldDir = (viewToWorld * vec4(viewDir, 0.0)).xyz;

    // first, calculate distance until hard surface and shade it
    float dist = shortestDistanceToSurface(eye, worldDir, MIN_DIST, MAX_DIST);
    if (dist > MAX_DIST - EPSILON) {
        // Didn't hit anything
        gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0);
    } else {
        // The closest point on the surface to the eyepoint along the view ray
        vec3 p = eye + dist * worldDir;

        vec3 color = illumination(p, eye, vec3(0.75, 0.5, 0.5), blueRand);

        gl_FragColor = vec4(color, 1.0);
    }

    // then calculate volumetric samples (contribution from passing through air particles)
    float volD = VOL_STEP_D * blueRand.z;
    vec4 volAccumulator = vec4(0.0, 0.0, 0.0, 1.0);
    vec3 volAlbedo = vec3(1.0, 1.0, 1.0);
    for (int volStep = 0; volStep < 20; volStep++) {
        float newVolD = min(dist, volD + VOL_STEP_D);
        float transmittance = exp(-VOL_AMOUNT * (newVolD - volD));
        volD = newVolD;

        vec3 volPos = eye + worldDir * volD;

        // add new scattering as occluded by existing accumulated opaqueness
        vec4 nRand = texture(blueNoise, blueRand.xy + float(volStep) / 20.0);
        vec3 scatterNormal = normalize(nRand.xyz);

        vec3 scatterLighting = contribForSun(volPos, scatterNormal, volAlbedo);
        scatterLighting += contribForBounce(volPos, scatterNormal, volAlbedo, nRand);

        vec3 scatterContrib = scatterLighting * (1.0 - transmittance);
        volAccumulator.rgb += scatterContrib * volAccumulator.a;

        // update total opaqueness
        volAccumulator.a *= transmittance;

        // bail if reached hard surface
        if (volD >= dist) {
            break;
        }

        // bail early if further scattering would be fully occluded
        if (volAccumulator.a < 0.003) {
            volAccumulator.a = 0.0;
            break;
        }
    }

    // occlude hard surface by the volume samples
    gl_FragColor.rgb *= volAccumulator.a;

    // add the luminance from volume samples
    gl_FragColor.rgb += volAccumulator.rgb;
    // @todo tone curve

}
