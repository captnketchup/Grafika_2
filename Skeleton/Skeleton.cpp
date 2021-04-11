//=============================================================================================
// Mintaprogram: Zöld háromszög. Ervenyes 2019. osztol.
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
// Nev    : 
// Neptun : 
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

void vec3Print(std::string name, vec3 vector);

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

struct Material {
	vec3 ka, kd, ks;
	float shininess;
	bool isReflective;
	Material(vec3 _kd, vec3 _ks, float _shininess, bool _isReflective) : ka(_kd *M_PI), kd(_kd), ks(_ks), shininess(_shininess), isReflective(_isReflective) {};
};

struct Hit {
	float t;
	vec3 position, normal;
	Material *material;
	Hit() { t = -1; }
};

struct Ray {
	vec3 start, dir;
	Ray(vec3 _start, vec3 _dir) {
		start = _start; dir = normalize(_dir);
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
	Dodecahedron() {
		//TODO: ha nagyon fucked a tükrözõdés akkor ezeket kell baszni
		vec3 kd(3.5f, 0.5f, 0.5f), ks(500, 0, 0);
		material = new Material(kd, ks, 5000, false);
		material->isReflective = false;
	}

	std::pair<vec3, vec3> getObjectPlane(int faceIndex) {
		vec3 p1 = vertices[faces[faceIndex][0] - 1];
		vec3 p2 = vertices[faces[faceIndex][1] - 1];
		vec3 p3 = vertices[faces[faceIndex][2] - 1];

		vec3 normal = cross(p2 - p1, p3 - p1);
		//WARN: idk a p2-p1 jó-e, eredetileg p1 volt csak de az nem logikus ????
		if (dot(p1, normal) < 0)	//ha kifelé mutat meginvertáljuk
			normal = -normal;
		//TODO: szkél??
		return std::make_pair(p1, normal);
	}

	Hit intersect(const Ray &ray, Hit hit) {
		for (int i = 0; i < faces.size(); i++) {
			std::pair<vec3, vec3> planePair = getObjectPlane(i);
			vec3 p1 = planePair.first;	//first point of face
			vec3 normal = planePair.second;	//the normal of the plane

			float ti;	//distance on ray from start 'till first intersection
			if (abs(dot(normal, ray.dir)) > EPSILON) {
				ti = dot(p1 - ray.start, normal) / dot(normal, ray.dir);
			}
			else {
				ti = -1.0f;
			}

			if (ti <= EPSILON || (ti > hit.t && hit.t > 0)) continue;	//is current ti closer than hit.t?
			vec3 pintersect = ray.start + ray.dir * ti;

			//now we check if intersection is outside of the face or not
			bool outside = false;
			for (int j = 0; j < faces.size(); j++) {
				if (i == j) continue;
				std::pair<vec3, vec3> otherPlanePair = getObjectPlane(j);
				vec3 otherPoint = otherPlanePair.first;
				vec3 otherNormal = otherPlanePair.second;
				//printf("normal: %3.5f, %3.5f, %3.5f\npintersect: %3.5f, %3.5f, 3.5f\nother point: %3.5f, %3.5f, %3.5f\n", normal.x, normal.y, normal.z, pintersect.x, pintersect.y, pintersect.z, otherPoint.x, otherPoint.y, otherPoint.z);
				if (dot(otherNormal, pintersect - otherPoint) > 0) {	//checks if normalvector is pointing inwards
					//printf("entered if(dot(normal, pintersect - otherPoint > 0))\n");
					outside = true;
					break;
				}
			}
			if (!outside) {
				hit.t = ti;
				hit.position = pintersect;
				hit.normal = normalize(normal);
				hit.material = material;
				//printf("intersect");
			}
		}
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
	//vec3 position;
	vec3 direction;
	vec3 Le;
	Light(vec3 _direction, vec3 _Le) : Le(_Le) {
		direction = normalize(_direction);
	};
};

class Scene {
	Dodecahedron dodecahedron;
	std::vector<Light *> lights;
	vec3 La = { 0.4f, 0.4f, 0.4f };
	//TODO: ellipsoid

public:
	Camera camera;
	void build() {
		camera = Camera();
		camera.set(vec3(0.5f, 0.5f, 0.5f), vec3(0.0f, 0.0f, 0.0f));
		dodecahedron = Dodecahedron();
		vec3 Le = { 2.0f, 2.0f, 2.0f };
		vec3 lightDirection = { 5.0f, 5.0f, 1.0f };
		lights.push_back(new Light(lightDirection, Le));
		lights.push_back(new Light(lightDirection * (-1), Le));
		//TODO: ellipsoid
	}

	Hit firstIntersect(Ray ray) {
		Hit bestHit = Hit();
		//TODO: Ellipsoid ray megvizsgálása

		bestHit = dodecahedron.intersect(ray, bestHit);
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

	bool shadowIntersect(Ray ray) {	// for directional lights
		//TODO: ezt szépítsd meg úristen de ronda
		Hit hit;
		if (dodecahedron.intersect(ray, hit).t > 0) return true;
		return false;
	}

	int maxDepth = 5;

	vec3 trace(Ray ray, int depth = 0) {
		Hit hit = firstIntersect(ray);
		if (hit.t < 0) return La;
		//return vec3(1.0f, 0.0f, 0.0f);
		vec3 outRadiance(0, 0, 0);		// the radiancy(kinda color) of a given point where the ray intersects
		outRadiance = outRadiance + La; // we add the ambient light to it
		if (!hit.material->isReflective) { //if material is rough
			for (Light *light : lights) {
				//TODO: epsilon
				//Ray shadowRay(hit.position + hit.normal * EPSILON, light->position - hit.position);
				Ray shadowRay(hit.position + hit.normal * EPSILON, light->direction);
				float cosTheta = dot(hit.normal, light->direction);		// is used to determine whether light is coming behind from the object
				//Hit shadowHit = firstIntersect(shadowRay);
				// if there's no object between the light source and the given point
				//if (cosTheta > 0 && (hit.t < 0 || shadowHit.t > length(light->position - hit.position))) {
				if (cosTheta > 0 && !shadowIntersect(shadowRay)) {
					outRadiance = outRadiance + light->Le * hit.material->kd * cosTheta;
					vec3 halfway = normalize(-ray.dir + light->direction);
					float cosDelta = dot(hit.normal, halfway);
					if (cosDelta > 0) outRadiance = outRadiance + light->Le * hit.material->ks * powf(cosDelta, hit.material->shininess);
				}
			}
		}
		else {
			//TODO: trace rekurzió
		}
		// maybe ray.weight???????????????????
		return outRadiance;
	}
} scene;

class FullScreenTexturedQuad {
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