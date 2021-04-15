//=============================================================================================
// Mintaprogram: Zold haromszog. Ervenyes 2019. osztol.
//
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!! 
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Kovari Daniel Mate
// Neptun : JDP18V
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
#include "framework.h"

// vertex shader in GLSL
const char *vertexSource = R"(
	#version 330
    precision highp float;

	layout(location = 0) in vec2 cVertexPosition;	// Attrib Array 0
	out vec2 texcoord;

	void main() {
		texcoord = (cVertexPosition + vec2(1, 1))/2;							// -1,1 to 0,1
		gl_Position = vec4(cVertexPosition.x, cVertexPosition.y, 0, 1); 		// transform to clipping space
	}
)";

// fragment shader in GLSL
const char *fragmentSource = R"(
	#version 330
    precision highp float;

	uniform sampler2D textureUnit;
	in  vec2 texcoord;			// interpolated texture coordinates
	out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation

	void main() {
		fragmentColor = texture(textureUnit, texcoord); 
	}
)";


GPUProgram gpuProgram; 
unsigned int vao;	   
const float EPSILON = 0.01f;
bool isMouseDown = false;
int mouseX;
int mouseY;

template<typename T1, typename T2>
class Pair {
public:
	T1 first;
	T2 second;
	Pair() {}
	Pair(T1 _first, T2 _second) {
		first = _first;
		second = _second;
	}
};

Pair<float, float> quadraticEq(float a, float b, float c);
bool hasRoot(float a, float b, float c);

struct Material {
	vec3 ka, kd, ks;
	float shininess;
	Material(vec3 _kd, vec3 _ks, float _shininess, bool _isReflective) : ka(0.02f, 0.02f, 0.02f), kd(_kd), ks(_ks), shininess(_shininess) {};
};

struct Hit {
	float t;
	vec3 position, normal;
	Material *material = nullptr;
	bool isReflective = false;
	bool isPortal = false;
	Hit() { t = -1; }
};

struct Ray {
	vec3 start, dir;
	Ray(vec3 _start, vec3 _dir) {
		start = _start; dir = normalize(_dir);
	}
	void RayPrint() const {
		printf("Start: x: %3.5f\ty: %3.5f\tz: %3.5f\nDir: x: %3.5f\ty: %3.5f\tz: %3.5f\n", start.x, start.y, start.z, dir.x, dir.y, dir.z);
	}
};

class Intersectable {
protected:
	Material *material = nullptr;
public:
	virtual Hit intersect(const Ray &ray) = 0;
};

class Dodecahedron : public Intersectable {
	std::vector<vec3> vertices = {
		{0.0f, 0.618f, 1.618f},
		{0.0f, -0.618, 1.618f},
		{0.0f, -0.618f, -1.618f},
		{0.0f, 0.618f, -1.618f},
		{1.618f, 0.0f, 0.618f},
		{-1.618f, 0.0f, 0.618f},
		{-1.618f, 0.0f, -0.618f},
		{1.618f, 0.0f, -0.618f},
		{0.618f, 1.618f, 0.0f},
		{-0.618f, 1.618f, 0.0f},
		{-0.618f, -1.618f, 0.0f},
		{0.618f, -1.618f, 0.0f},
		{1.0f, 1.0f, 1.0f},
		{-1.0f, 1.0f, 1.0f},
		{-1.0f, -1.0f, 1.0f},
		{1.0f, -1.0f, 1.0f},
		{1.0f, -1.0f, -1.0f},
		{1.0f, 1.0f, -1.0f},
		{-1.0f, 1.0f, -1.0f},
		{-1.0f, -1.0f, -1.0f}
	};

	std::vector<std::vector<int>> faces = {
			{1,	 2,  16, 5,  13},
			{1,  13, 9,  10, 14},
			{1,  14, 6,  15, 2},
			{2,  15, 11, 12, 16},
			{3,  4,  18, 8,  17},
			{3,  17, 12, 11, 20},
			{3,  20, 7,  19, 4},
			{19, 10, 9,  18, 4},
			{16, 12, 17, 8,  5},
			{5,  8,  18, 9,  13},
			{14, 10, 19, 7,  6},
			{6,  7,  20, 11, 15}
	};

public:

	Dodecahedron() {
		vec3 kd(0.592f, 0.0f, 0.639f), ks(0.8f, 0.2f, 0.4f);		
		material = new Material(kd, ks, 10, false);
	}

	Pair<vec3, vec3> getObjectPlane(int faceIndex) {
		vec3 p1 = vertices[faces[faceIndex][0] - 1];
		vec3 p2 = vertices[faces[faceIndex][1] - 1];
		vec3 p3 = vertices[faces[faceIndex][2] - 1];

		vec3 normal = cross(p2 - p1, p3 - p1);
		if (dot(p1, normal) < 0)	
			normal = -normal;
		return Pair<vec3, vec3>(p1, normal);
	}

	float distanceFromPlane(vec3 point, vec3 planePoint, vec3 planeNormal) {
		return fabs(dot(point - planePoint, planeNormal));
	}

	Hit intersect(const Ray &ray) {
		Hit hit = Hit();
		for (int i = 0; i < faces.size(); i++) {
			Pair<vec3, vec3> planePair = getObjectPlane(i);
			vec3 planePoint = planePair.first;	
			vec3 planeNormal = planePair.second;	

			float ti;	
			if (fabs(dot(planeNormal, ray.dir)) > EPSILON) {
				ti = dot(planePoint - ray.start, planeNormal) / dot(planeNormal, ray.dir);
			}
			else {
				ti = -1.0f;
			}

			if (ti <= EPSILON) continue;	
			vec3 pintersect = ray.start + ray.dir * ti;

			bool outside = false;
			bool reflective = true;
			for (int j = 0; j < faces.size(); j++) {
				if (i == j) continue;
				Pair<vec3, vec3> otherPlanePair = getObjectPlane(j);
				vec3 otherPlanePoint = otherPlanePair.first;
				vec3 otherPlaneNormal = otherPlanePair.second;
				if (dot(otherPlaneNormal, pintersect - otherPlanePoint) > 0) {	
					outside = true;
					break;
				}
				if (distanceFromPlane(pintersect, otherPlanePoint, otherPlaneNormal) < 0.1f) {
					reflective = false;
				}
			}
			if (!outside) {
				hit.t = ti;
				hit.position = pintersect;
				hit.normal = normalize(planeNormal);
				hit.material = material;
				hit.isReflective = reflective;
			}
		}
		return hit;

	}
};

class Paraboloid : public Intersectable {
	float expA = 0.8f, expB = 0.8f, expC = 0.1f;
	float radius = 0.3f;

public:
	vec3 n = { 0.17f, 0.35f, 1.5f };
	vec3 kappa = { 3.1f, 2.7f, 1.9f };

	Paraboloid() {
		vec3 kd(1.0f, 0.945f, 0.360f), ks(1.0f, 1.0f, 1.0f);		

		material = new Material(kd, ks, 50, true);
	}
	Hit intersect(const Ray &ray) {
		float a = expA * ray.dir.x * ray.dir.x +
			expB * ray.dir.y * ray.dir.y;
		float b = 2 * expA * ray.start.x * ray.dir.x +
			expB * 2 * ray.start.y * ray.dir.y -
			expC * ray.dir.z;
		float c = expA * ray.start.x * ray.start.x +
			expB * ray.start.y * ray.start.y -
			expC * ray.start.z;

		if (!hasRoot(a, b, c)) return Hit();

		Pair<float, float> tIntersect = quadraticEq(a, b, c);

		float t1 = tIntersect.first;
		float t2 = tIntersect.second;

		vec3 tempPosition1 = ray.start + ray.dir * t1;
		vec3 tempPosition2 = ray.start + ray.dir * t2;

		float tTest = -1.0;

		if (dot(tempPosition1, tempPosition1) > (radius * radius))
			t1 = -1.0f;		

		if (dot(tempPosition2, tempPosition2) > (radius * radius))
			t2 = -1.0f;		


		if (t1 > 0) tTest = t1;		

		if (t2 > 0 && t2 < tTest) {
			tTest = t2;
		}

		if (tTest <= 0) return Hit();		

		Hit hit = Hit();
		hit.t = tTest;
		hit.position = ray.start + ray.dir * hit.t;

		vec3 focusPoint = { 0.0f, hit.position.y / (4 * (hit.position.x * hit.position.x + hit.position.z * hit.position.z)), 0.0f };
		vec3 paraboloidA = hit.position - focusPoint;
		vec3 paraboloidB = length(paraboloidA) * vec3(0.0f, 1.0f, 0.0f);
		vec3 normal = (-1) * normalize(paraboloidA + paraboloidB);
		hit.normal = normal;
		hit.material = material;
		return hit;
	}
};


class Camera {
	vec3 eye, lookat, right, up;
	vec3 vup = { 0.0f, 1.0f, 0.0f };
	float fov = 90.0f * M_PI / 180.0f;
public:
	void set(vec3 _eye, vec3 _lookat) {
		eye = _eye;
		lookat = _lookat;
		vec3 w = eye - lookat;
		float focus = length(w);
		right = normalize(cross(vup, w)) * focus * tanf(fov / 2);
		up = normalize(cross(w, right)) * focus * tanf(fov / 2);
	}
	Ray getRay(int X, int Y) {
		vec3 dir = lookat + right * (2.0f * (X + 0.5f) / windowWidth - 1) + up * (2.0f * (Y + 0.5f) / windowHeight - 1) - eye;
		return Ray(eye, dir);
	}
	void Animate(float t)
	{
		float r = sqrtf(eye.x * eye.x + eye.y * eye.y);
		eye = vec3(r * cosf(t) + lookat.x, r * sinf(t) + lookat.y, eye.z);
		set(eye, lookat);
	}
};

struct Light {
	vec3 position;
	vec3 Le;
	Light(vec3 _position, vec3 _Le) : position(_position), Le(_Le) {};
};

vec3 operator/(vec3 lhs, vec3 rhs) {
	return vec3(lhs.x / rhs.x, lhs.y / rhs.y, lhs.z / rhs.z);
}

class Scene {
	Dodecahedron dodecahedron;
	Paraboloid paraboloid;
	std::vector<Light *> lights;
	vec3 La = { 0.584f, 0.827f, 0.933f };

	vec3 reflect(vec3 inDirectionVec, vec3 normal) {
		return inDirectionVec - normal * dot(normal, inDirectionVec) * 2.0f;
	}

	vec3 Fresnel(vec3 inDirectionVec, vec3 normal) {
		float cosa = -dot(inDirectionVec, normal);
		vec3 one(1, 1, 1);
		vec3 F0 = vec3();
		vec3 n = paraboloid.n;
		vec3 kappa = paraboloid.kappa;
		F0 = ((n - one) * (n - one) + (kappa * kappa)) /
			((n + one) * (n + one) + (kappa * kappa));
		return F0 + (one - F0) * powf( cosa, 5);		
	}

public:
	Camera camera;
	void build() {
		camera = Camera();
		camera.set(vec3(0.5f, 0.5f, 0.5f), vec3(0.0f, 0.0f, 0.0f));
		dodecahedron = Dodecahedron();
		paraboloid = Paraboloid();
		vec3 Le = { 1.0f, 1.0f, 1.0f };
		vec3 lightPosition = { 0.0f, 0.0f, 0.0f };			
		lights.push_back(new Light(lightPosition, Le));
	}

	void render(std::vector<vec4> &image) {
		vec4 black = vec4(1.0f, 1.0f, 1.0f, 1.0f);
		vec4 white = vec4(0.0f, 0.0f, 0.0f, 1.0f);
		for (int Y = 0; Y < windowHeight; Y++) {
#pragma omp parallel for
			for (int X = 0; X < windowWidth; X++) {
				vec3 color = trace(camera.getRay(X, Y));
				image[Y * windowWidth + X] = vec4(color.x, color.y, color.z, 1);
			}
		}
	}

	bool shadowIntersect(Ray ray) {
		Hit hit;
		if (dodecahedron.intersect(ray).t > 0) return true;
		return false;
	}

	int maxDepth = 5;

	vec3 trace(Ray ray, int depth = 0) {
		if (depth > maxDepth) return La;

		Hit paraboloidHit = paraboloid.intersect(ray);
		Hit dodecaHit = dodecahedron.intersect(ray);
		vec3 outRadiance(0.0f, 0.0f, 0.0f);		

		if (dot(ray.dir, paraboloidHit.normal) < 0) paraboloidHit.normal = paraboloidHit.normal * (-1);
		if (dot(ray.dir, dodecaHit.normal) < 0) dodecaHit.normal = dodecaHit.normal * (-1);

		if (paraboloidHit.t >= 0 && paraboloidHit.t < dodecaHit.t) {
			vec3 reflectionDir = reflect(ray.dir, paraboloidHit.normal);
			Ray reflectRay(paraboloidHit.position - paraboloidHit.normal * EPSILON, reflectionDir);		
			outRadiance = outRadiance + trace(reflectRay, depth + 1) * Fresnel(ray.dir, paraboloidHit.normal);
		}
		else if (dodecaHit.t >= 0) {
			if (!dodecaHit.isReflective) { 
				outRadiance = dodecaHit.material->ka * La;
				for (Light *light : lights) {
					vec3 lightDirection = dodecaHit.position - light->position;
					Ray shadowRay(dodecaHit.position + dodecaHit.normal * EPSILON, lightDirection);
					float cosTheta = dot(dodecaHit.normal, lightDirection);	
					if (cosTheta > 0 && !shadowIntersect(shadowRay)) {
						outRadiance = outRadiance + light->Le * dodecaHit.material->kd * cosTheta;
						vec3 halfway = normalize(-ray.dir + lightDirection);
						float cosDelta = dot(dodecaHit.normal, halfway);
						if (cosDelta > 0) outRadiance = outRadiance + light->Le * dodecaHit.material->ks * powf(cosDelta, dodecaHit.material->shininess);
					}
				}
			}
			else {	
				vec3 reflectionDir = reflect(ray.dir, dodecaHit.normal);
				vec4 tempReflDir = { reflectionDir.x, reflectionDir.y, reflectionDir.z, 0.0f };
				mat4 rotationMat = RotationMatrix(72.0f * M_PI/180.0f, dodecaHit.normal);
				tempReflDir = tempReflDir * rotationMat;
				reflectionDir.x = tempReflDir.x;
				reflectionDir.y = tempReflDir.y;
				reflectionDir.z = tempReflDir.z;
				vec4 tempPosition = { dodecaHit.position.x, dodecaHit.position.y, dodecaHit.position.z, 0.0f };
				tempPosition = tempPosition * rotationMat;
				dodecaHit.position.x = tempPosition.x;
				dodecaHit.position.y = tempPosition.y;
				dodecaHit.position.z = tempPosition.z;

				Ray reflectRay(dodecaHit.position - dodecaHit.normal * EPSILON, reflectionDir);

				outRadiance = outRadiance + trace(reflectRay, depth + 1);
			}
		}
		else {
			return La;
		}
		return outRadiance;
	}
} scene;

class FullScreenTexturedQuad {
	unsigned int vao;	
	Texture texture;
public:
	FullScreenTexturedQuad(int windowWidth, int windowHeight, std::vector<vec4> &image)
		: texture(windowWidth, windowHeight, image)
	{
		glGenVertexArrays(1, &vao);	
		glBindVertexArray(vao);		

		unsigned int vbo;		
		glGenBuffers(1, &vbo);

		glBindBuffer(GL_ARRAY_BUFFER, vbo); 
		float vertexCoords[] = { -1, -1,  1, -1,  1, 1,  -1, 1 };	
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW);	  
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);     
	}

	void Draw() {
		glBindVertexArray(vao);	
		gpuProgram.setUniform(texture, "textureUnit");
		glDrawArrays(GL_TRIANGLE_FAN, 0, 4);	
	}
};

FullScreenTexturedQuad *fullScreenTexturedQuad;

void sceneRender() {
	
	std::vector<vec4> image(windowWidth * windowHeight);

	scene.render(image);

	
	fullScreenTexturedQuad = new FullScreenTexturedQuad(windowWidth, windowHeight, image);
}


void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	Dodecahedron dodecahedron;
	
	scene = Scene();
	scene.build();

	sceneRender();

	
	gpuProgram.create(vertexSource, fragmentSource, "outColor");
}


void onDisplay() {
	fullScreenTexturedQuad->Draw();
	glutSwapBuffers();
}

void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == 'd') glutPostRedisplay();       
}


void onKeyboardUp(unsigned char key, int pX, int pY) {
}


void onMouseMotion(int pX, int pY) {	

	float cX = 2.0f * pX / windowWidth - 1;
	float cY = 1.0f - 2.0f * pY / windowHeight;
}


void onMouse(int button, int state, int pX, int pY) {
	float cX = 2.0f * pX / windowWidth - 1;	
	float cY = 1.0f - 2.0f * pY / windowHeight;

	mouseX = pX;
	mouseY = pY;
	if (state == GLUT_DOWN && button == GLUT_LEFT_BUTTON) {
		isMouseDown = true;
	}
	else isMouseDown = false;

	char *buttonStat;
	switch (state) {
		case GLUT_DOWN: buttonStat = "pressed"; break;
		case GLUT_UP:   buttonStat = "released"; break;
	}

	switch (button) {
		case GLUT_LEFT_BUTTON:   printf("Left button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY);   break;
		case GLUT_MIDDLE_BUTTON: printf("Middle button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY); break;
		case GLUT_RIGHT_BUTTON:  printf("Right button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY);  break;
	}
}


void onIdle() {
	float time = glutGet(GLUT_ELAPSED_TIME); 
	scene.camera.Animate(time / 5000);

	sceneRender();
	glutPostRedisplay();
}

bool hasRoot(float a, float b, float c) {
	return b * b - 4.0f * a * c > 0;
}

Pair<float, float> quadraticEq(float a, float b, float c) {
	float t1, t2 = 0.0f;

	t1 = ((-1) * b + sqrtf(b * b - 4 * a * c)) / (2 * a);
	t2 = ((-1) * b - sqrtf(b * b - 4 * a * c)) / (2 * a);

	Pair<float, float> tPair(t1, t2);
	return tPair;
}