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

//// vertex shader in GLSL: It is a Raw string (C++11) since it contains new line characters
//const char *const vertexSource = R"(
//	#version 330				// Shader 3.3
//	precision highp float;		// normal floats, makes no difference on desktop computers
//
//	uniform mat4 MVP;			// uniform variable, the Model-View-Projection transformation matrix
//	layout(location = 0) in vec2 vp;	// Varying input: vp = vertex position is expected in attrib array 0
//
//	void main() {
//		gl_Position = vec4(vp.x, vp.y, 0, 1) * MVP;		// transform vp from modeling space to normalized device space
//	}
//)";
//
//// fragment shader in GLSL
//const char *const fragmentSource = R"(
//	#version 330			// Shader 3.3
//	precision highp float;	// normal floats, makes no difference on desktop computers
//	
//	uniform vec3 color;		// uniform variable, the color of the primitive
//	out vec4 outColor;		// computed color of the current pixel
//
//	void main() {
//		outColor = vec4(color, 1);	// computed color is the color of the primitive
//	}
//)";


//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
//TODO : Pairt nem lehet használni !!!!!!  FONTOS
//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



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


GPUProgram gpuProgram; // vertex and fragment shaders
unsigned int vao;	   // virtual world on the GPU
const float EPSILON = 0.01f;

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

void vec3Print(std::string name, vec3 vector);
Pair<float, float> quadraticEq(float a, float b, float c);
bool hasRoot(float a, float b, float c);

struct Material {
	vec3 ka, kd, ks;
	float shininess;
	//Material(vec3 _kd, vec3 _ks, float _shininess, bool _isReflective) : ka(_kd *M_PI), kd(_kd), ks(_ks), shininess(_shininess) {};
	Material(vec3 _kd, vec3 _ks, float _shininess, bool _isReflective) : ka(0.02f, 0.02f, 0.02f), kd(_kd), ks(_ks), shininess(_shininess) {};
};

struct Hit {
	float t;
	vec3 position, normal;
	Material *material;
	bool isReflective;
	bool isPortal;
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
	Material *material;
public:
	virtual Hit intersect(const Ray &ray, Hit hit) = 0;
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
	vec3 n = vec3(0.01, 0.01, 0.01);
	//vec3 n = vec3(1,1,1);		//ezeket az egyik vidyaban láttam mint tökéletes tükrözõ anyag, de elég fuckedul néz ki mert tiszta köd
	//vec3 kappa = vec3(5,4,3);
	vec3 kappa = vec3(10.0, 10.0, 10.0);

	Dodecahedron() {
		//TODO: ha nagyon fucked a tükrözõdés akkor ezeket kell baszni
		//vec3 kd(0.5f, 0.05f, 1.5f), ks(500.0f, 100.0f, 100.0f);
		vec3 kd(0.592f, 0.0f, 0.639f), ks(0.8f, 0.2f, 0.4f);		//ks hogy melyik színt hogy csillantja meg
		//material = new Material(kd, ks, 50, false);
		material = new Material(kd, ks, 10, false);
	}

	Pair<vec3, vec3> getObjectPlane(int faceIndex) {
		vec3 p1 = vertices[faces[faceIndex][0] - 1];
		vec3 p2 = vertices[faces[faceIndex][1] - 1];
		vec3 p3 = vertices[faces[faceIndex][2] - 1];

		vec3 normal = cross(p2 - p1, p3 - p1);
		if (dot(p1, normal) < 0)	//ha kifelé mutat meginvertáljuk
			normal = -normal;
		//TODO: szkél??
		return Pair<vec3, vec3>(p1, normal);
	}

	float distanceFromPlane(vec3 point, vec3 planePoint, vec3 planeNormal) {
		return abs(dot(point - planePoint, planeNormal));
	}

	Hit intersect(const Ray &ray, Hit hit) {
		for (int i = 0; i < faces.size(); i++) {
			Pair<vec3, vec3> planePair = getObjectPlane(i);
			vec3 planePoint = planePair.first;	//first point of face
			vec3 planeNormal = planePair.second;	//the normal of the plane

			float ti;	//distance on ray from start 'till first intersection
			if (abs(dot(planeNormal, ray.dir)) > EPSILON) {
				ti = dot(planePoint - ray.start, planeNormal) / dot(planeNormal, ray.dir);
			}
			else {
				ti = -1.0f;
			}

			if (ti <= EPSILON || (ti > hit.t && hit.t > 0)) continue;	//is current ti closer than hit.t?
			vec3 pintersect = ray.start + ray.dir * ti;

			//now we check if intersection is outside of the face or not
			bool outside = false;
			bool reflective = true;
			for (int j = 0; j < faces.size(); j++) {
				if (i == j) continue;
				Pair<vec3, vec3> otherPlanePair = getObjectPlane(j);
				vec3 otherPlanePoint = otherPlanePair.first;
				vec3 otherPlaneNormal = otherPlanePair.second;
				//printf("normal: %3.5f, %3.5f, %3.5f\npintersect: %3.5f, %3.5f, 3.5f\nother point: %3.5f, %3.5f, %3.5f\n", normal.x, normal.y, normal.z, pintersect.x, pintersect.y, pintersect.z, otherPoint.x, otherPoint.y, otherPoint.z);
				if (dot(otherPlaneNormal, pintersect - otherPlanePoint) > 0) {	//checks if normalvector is pointing inwards
					//printf("entered if(dot(normal, pintersect - otherPoint > 0))\n");
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
				//printf("intersect");
			}
		}
		return hit;

	}
};

class Paraboloid : public Intersectable {
	float expA = 0.8f, expB = 0.8f, expC = 0.1f;
	float radius = 0.3f;

public:
	// meg voltak adva ezek
	vec3 n = { 0.17f, 0.35f, 1.5f };
	vec3 kappa = { 3.1f, 2.7f, 1.9f };

	Paraboloid() {
		vec3 kd(1.0f, 0.945f, 0.360f), ks(1.0f, 1.0f, 1.0f);		//idk aranyszínû szerû

		material = new Material(kd, ks, 50, true);
	}
	Hit intersect(const Ray &ray, Hit hit) {
		//checks if intersects paraboloid
		float a = expA * ray.dir.x * ray.dir.x +
			expB * ray.dir.y * ray.dir.y;
		float b = 2 * expA * ray.start.x * ray.dir.x +
			expB * 2 * ray.start.y * ray.dir.y -
			expC * ray.dir.z;
		float c = expA * ray.start.x * ray.start.x +
			expB * ray.start.y * ray.start.y -
			expC * ray.start.z;

		if (!hasRoot(a, b, c)) return hit;

		Pair<float, float> tIntersect = quadraticEq(a, b, c);

		float t1 = tIntersect.first;
		float t2 = tIntersect.second;

		vec3 tempPosition1 = ray.start + ray.dir * t1;
		vec3 tempPosition2 = ray.start + ray.dir * t2;

		//if ((ray.dir.x + 0.57735) * (ray.dir.x + 0.57735) + (ray.dir.y + 0.57735) * (ray.dir.y + 0.57735) + (ray.dir.z + 0.57735) * (ray.dir.z + 0.57735) < 0.05) {
		//	vec3Print("tempPosition", tempPosition);
		//	ray.RayPrint();
		//}
		float tTest = -1.0;

		if (dot(tempPosition1, tempPosition1) > (radius * radius))
			t1 = -1.0f;		//t1 nem jo

		if (dot(tempPosition2, tempPosition2) > (radius * radius))
			t2 = -1.0f;		//t2 nem jo


		if (t1 > 0) tTest = t1;		//t1 ervenyes

		if (t2 > 0 && t2 < tTest) {
			tTest = t2;
		}

		if (tTest <= 0) return hit;		//idk hogy kell-e egyenlo

		hit.t = tTest;
		hit.position = ray.start + ray.dir * hit.t;

		//https://aggie.io/ysax3oxo0k
		vec3 focusPoint = { 0.0f, hit.position.y / (4 * (hit.position.x * hit.position.x + hit.position.z * hit.position.z)), 0.0f };
		vec3 paraboloidA = hit.position - focusPoint;
		vec3 paraboloidB = length(paraboloidA) * vec3(0.0f, 1.0f, 0.0f);
		vec3 normal = (-1) * normalize(paraboloidA + paraboloidB);
		hit.normal = normal;
		hit.material = material;
		return hit;
	}


};


//TODO: megérteni mi történik itt
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
		//rotates the camera around lookat ( = (0, 0, 0))
		float r = sqrtf(eye.x * eye.x + eye.y * eye.y);
		eye = vec3(r * cos(t) + lookat.x, r * sin(t) + lookat.y, eye.z);
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
	//vec3 La = { 0.4f, 0.4f, 0.4f };
	vec3 La = { 0.584f, 0.827f, 0.933f };

	// returns the reflected ray's direction vector
	vec3 reflect(vec3 inDirectionVec, vec3 normal) {
		return inDirectionVec - normal * dot(normal, inDirectionVec) * 2.0f;
	}

	vec3 Fresnel(vec3 inDirectionVec, vec3 normal) {
		float cosa = -dot(inDirectionVec, normal);
		vec3 one(1, 1, 1);
		vec3 F0 = vec3();
		vec3 n = paraboloid.n;
		vec3 kappa = paraboloid.kappa;
		F0 = ((n - one) * (n - one) + (kappa * kappa)) / ((n + one) * (n + one) + (kappa * kappa));
		/*F0.x = ((n.x - one.x) * (n.x - one.x) + kappa.x * kappa.x) / ((n.x + one.x) * (n.x + one.x) + kappa.x * kappa.x);
		F0.y = ((n.y - one.y) * (n.y - one.y) + kappa.y * kappa.y) / ((n.y + one.y) * (n.y + one.y) + kappa.y * kappa.y);
		F0.z = ((n.z - one.z) * (n.z - one.z) + kappa.z * kappa.z) / ((n.z + one.z) * (n.z + one.z) + kappa.z * kappa.z);*/
		return F0 + (one - F0) * pow(1 - cosa, 5);
	}

public:
	Camera camera;
	void build() {
		camera = Camera();
		camera.set(vec3(0.5f, 0.5f, 0.5f), vec3(0.0f, 0.0f, 0.0f));
		dodecahedron = Dodecahedron();
		paraboloid = Paraboloid();
		//vec3 Le = { 2.0f, 2.0f, 2.0f };
		vec3 Le = { 1.0f, 1.0f, 1.0f };
		vec3 lightPosition = { 0.0f, 0.0f, 0.0f };			// ez nem lesz jó mert itt lesz a cucc középen
		//vec3 lightPosition = { 0.2f, 0.2f, 0.2f };
		lights.push_back(new Light(lightPosition, Le));
		//lights.push_back(new Light(lightDirection * (-1), Le));
		//TODO: ellipsoid haha nem is az xDDDD
	}

	Hit firstIntersect(Ray ray) {
		Hit bestHit = Hit();
		//TODO: Ellipsoid ray megvizsgálása
		bestHit = dodecahedron.intersect(ray, bestHit);
		bestHit = paraboloid.intersect(ray, bestHit);
		if (dot(ray.dir, bestHit.normal) < 0) bestHit.normal = bestHit.normal * (-1);
		return bestHit;
	}

	void render(std::vector<vec4> &image) {
		vec4 black = vec4(1.0f, 1.0f, 1.0f, 1.0f);
		vec4 white = vec4(0.0f, 0.0f, 0.0f, 1.0f);
		for (int Y = 0; Y < windowHeight; Y++) {
#pragma omp parallel for
			for (int X = 0; X < windowWidth; X++) {
				vec3 color = trace(camera.getRay(X, Y));
				image[Y * windowWidth + X] = vec4(color.x, color.y, color.z, 1);
				//image[Y * windowWidth + X] = ((X + Y) % 2) ? black : white;
			}
		}
	}

	// checks if shot out light intersects with something
	bool shadowIntersect(Ray ray) {
		//TODO: ezt szépítsd meg úristen de ronda
		Hit hit;
		if (dodecahedron.intersect(ray, hit).t > 0) return true;
		return false;
	}

	int maxDepth = 5;

	vec3 trace(Ray ray, int depth = 0) {
		if (depth > maxDepth) return La;
		Hit hit = firstIntersect(ray);
		if (hit.t < 0) return La;
		//return vec3(1.0f, 0.0f, 0.0f);
		vec3 outRadiance(0, 0, 0);		// the radiancy(kinda color) of a given point where the ray intersects
		//outRadiance = outRadiance + La; // we add the ambient light to it
		outRadiance = hit.material->ka * La;		// ezt találtam a rekurzív raytracelõs videóban 07:28, ekcsölli jól néz ki
		if (!hit.isReflective) { //if material is rough
			for (Light *light : lights) {
				//TODO: epsilon
				vec3 lightDirection = hit.position - light->position;
				Ray shadowRay(hit.position + hit.normal * EPSILON, lightDirection);
				//Ray shadowRay(hit.position + hit.normal * EPSILON, light->direction);
				float cosTheta = dot(hit.normal, lightDirection);		// is used to determine whether light is coming behind from the object
				//Hit shadowHit = firstIntersect(shadowRay);
				// if there's no object between the light source and the given point
				//if (cosTheta > 0 && (hit.t < 0 || shadowHit.t > length(light->position - hit.position))) {
				if (cosTheta > 0 && !shadowIntersect(shadowRay)) {
					outRadiance = outRadiance + light->Le * hit.material->kd * cosTheta;
					vec3 halfway = normalize(-ray.dir + lightDirection);
					float cosDelta = dot(hit.normal, halfway);
					if (cosDelta > 0) outRadiance = outRadiance + light->Le * hit.material->ks * powf(cosDelta, hit.material->shininess);
				}
			}
		}
		else {
			vec3 reflectionDir = reflect(ray.dir, hit.normal);
			Ray reflectRay(hit.position - hit.normal * EPSILON, reflectionDir);

			outRadiance = outRadiance + trace(reflectRay, depth + 1) * Fresnel(ray.dir, hit.normal);


		}
		return outRadiance;
	}
} scene;

class FullScreenTexturedQuad {
	//destruktor????
	unsigned int vao;	// vertex array object id and texture id
	Texture texture;
public:
	FullScreenTexturedQuad(int windowWidth, int windowHeight, std::vector<vec4> &image)
		: texture(windowWidth, windowHeight, image)
	{
		glGenVertexArrays(1, &vao);	// create 1 vertex array object
		glBindVertexArray(vao);		// make it active

		unsigned int vbo;		// vertex buffer objects
		glGenBuffers(1, &vbo);	// Generate 1 vertex buffer objects

		// vertex coordinates: vbo0 -> Attrib Array 0 -> vertexPosition of the vertex shader
		glBindBuffer(GL_ARRAY_BUFFER, vbo); // make it active, it is an array
		float vertexCoords[] = { -1, -1,  1, -1,  1, 1,  -1, 1 };	// two triangles forming a quad
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW);	   // copy to that part of the memory which is not modified 
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);     // stride and offset: it is tightly packed
	}

	void Draw() {
		glBindVertexArray(vao);	// make the vao and its vbos active playing the role of the data source
		gpuProgram.setUniform(texture, "textureUnit");
		glDrawArrays(GL_TRIANGLE_FAN, 0, 4);	// draw two triangles forming a quad
	}
};

FullScreenTexturedQuad *fullScreenTexturedQuad;

void sceneRender() {
	// this will contain the pixel data
	std::vector<vec4> image(windowWidth * windowHeight);

	scene.render(image);

	// copy image to GPU as a texture
	fullScreenTexturedQuad = new FullScreenTexturedQuad(windowWidth, windowHeight, image);
}

// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	Dodecahedron dodecahedron();
	//TODO: create scene
	scene = Scene();
	scene.build();

	sceneRender();

	// create program for the GPU
	gpuProgram.create(vertexSource, fragmentSource, "outColor");
}

// Window has become invalid: Redraw
void onDisplay() {
	fullScreenTexturedQuad->Draw();
	glutSwapBuffers(); // exchange buffers for double buffering
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == 'd') glutPostRedisplay();         // if d, invalidate display, i.e. redraw
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {	// pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
	// Convert to normalized device space
	float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
	float cY = 1.0f - 2.0f * pY / windowHeight;
	printf("Mouse moved to (%3.2f, %3.2f)\n", cX, cY);
}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) { // pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
	// Convert to normalized device space
	float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
	float cY = 1.0f - 2.0f * pY / windowHeight;

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

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	float time = glutGet(GLUT_ELAPSED_TIME); // elapsed time since the start of the program
	scene.camera.Animate(time / 5000);

	sceneRender();
	glutPostRedisplay();
}


void vec3Print(std::string name, vec3 vector) {
	printf("%s: X: %3.2f, Y: %3.2f, Z: %3.2f\n", name.c_str(), vector.x, vector.y, vector.z);
}

bool hasRoot(float a, float b, float c) {
	return b * b - 4.0 * a * c > 0;
}

Pair<float, float> quadraticEq(float a, float b, float c) {
	float t1, t2 = 0.0f;

	t1 = ((-1) * b + sqrtf(b * b - 4 * a * c)) / (2 * a);
	t2 = ((-1) * b - sqrtf(b * b - 4 * a * c)) / (2 * a);

	Pair<float, float> tPair(t1, t2);
	return tPair;
}
