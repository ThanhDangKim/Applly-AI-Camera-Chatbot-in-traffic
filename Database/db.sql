--
-- PostgreSQL database dump
--

-- Dumped from database version 17.4
-- Dumped by pg_dump version 17.5

-- Started on 2025-06-13 12:17:32

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET transaction_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- TOC entry 224 (class 1259 OID 16453)
-- Name: areas; Type: TABLE; Schema: public; Owner: traffic_app_user
--

CREATE TABLE public.areas (
    id integer NOT NULL,
    name character varying(100) NOT NULL,
    description text
);


ALTER TABLE public.areas OWNER TO traffic_app_user;

--
-- TOC entry 223 (class 1259 OID 16452)
-- Name: areas_id_seq; Type: SEQUENCE; Schema: public; Owner: traffic_app_user
--

CREATE SEQUENCE public.areas_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.areas_id_seq OWNER TO traffic_app_user;

--
-- TOC entry 4893 (class 0 OID 0)
-- Dependencies: 223
-- Name: areas_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: traffic_app_user
--

ALTER SEQUENCE public.areas_id_seq OWNED BY public.areas.id;


--
-- TOC entry 222 (class 1259 OID 16437)
-- Name: avg_speeds; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.avg_speeds (
    id integer NOT NULL,
    camera_id integer,
    date date NOT NULL,
    time_slot smallint NOT NULL,
    average_speed double precision,
    CONSTRAINT avg_speeds_average_speed_check CHECK ((average_speed >= (0)::double precision)),
    CONSTRAINT avg_speeds_time_slot_check CHECK (((time_slot >= 0) AND (time_slot <= 47)))
);


ALTER TABLE public.avg_speeds OWNER TO postgres;

--
-- TOC entry 221 (class 1259 OID 16436)
-- Name: avg_speeds_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.avg_speeds_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.avg_speeds_id_seq OWNER TO postgres;

--
-- TOC entry 4895 (class 0 OID 0)
-- Dependencies: 221
-- Name: avg_speeds_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.avg_speeds_id_seq OWNED BY public.avg_speeds.id;


--
-- TOC entry 226 (class 1259 OID 16462)
-- Name: camera_area; Type: TABLE; Schema: public; Owner: traffic_app_user
--

CREATE TABLE public.camera_area (
    id integer NOT NULL,
    camera_id integer,
    area_id integer,
    location_detail text
);


ALTER TABLE public.camera_area OWNER TO traffic_app_user;

--
-- TOC entry 225 (class 1259 OID 16461)
-- Name: camera_area_id_seq; Type: SEQUENCE; Schema: public; Owner: traffic_app_user
--

CREATE SEQUENCE public.camera_area_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.camera_area_id_seq OWNER TO traffic_app_user;

--
-- TOC entry 4896 (class 0 OID 0)
-- Dependencies: 225
-- Name: camera_area_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: traffic_app_user
--

ALTER SEQUENCE public.camera_area_id_seq OWNED BY public.camera_area.id;


--
-- TOC entry 220 (class 1259 OID 16404)
-- Name: cameras; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.cameras (
    id integer NOT NULL,
    name character varying(100) NOT NULL,
    location text,
    installed_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP
);


ALTER TABLE public.cameras OWNER TO postgres;

--
-- TOC entry 219 (class 1259 OID 16403)
-- Name: cameras_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.cameras_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.cameras_id_seq OWNER TO postgres;

--
-- TOC entry 4898 (class 0 OID 0)
-- Dependencies: 219
-- Name: cameras_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.cameras_id_seq OWNED BY public.cameras.id;


--
-- TOC entry 228 (class 1259 OID 16499)
-- Name: daily_traffic_summary; Type: TABLE; Schema: public; Owner: traffic_app_user
--

CREATE TABLE public.daily_traffic_summary (
    id integer NOT NULL,
    camera_id integer NOT NULL,
    date date NOT NULL,
    total_vehicle_count integer,
    avg_speed double precision,
    peak_time_slot smallint,
    direction_with_most_traffic character varying(10),
    CONSTRAINT daily_traffic_summary_avg_speed_check CHECK ((avg_speed >= (0)::double precision)),
    CONSTRAINT daily_traffic_summary_peak_time_slot_check CHECK (((peak_time_slot >= 0) AND (peak_time_slot <= 47))),
    CONSTRAINT daily_traffic_summary_total_vehicle_count_check CHECK ((total_vehicle_count >= 0))
);


ALTER TABLE public.daily_traffic_summary OWNER TO traffic_app_user;

--
-- TOC entry 227 (class 1259 OID 16498)
-- Name: daily_traffic_summary_id_seq; Type: SEQUENCE; Schema: public; Owner: traffic_app_user
--

CREATE SEQUENCE public.daily_traffic_summary_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.daily_traffic_summary_id_seq OWNER TO traffic_app_user;

--
-- TOC entry 4899 (class 0 OID 0)
-- Dependencies: 227
-- Name: daily_traffic_summary_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: traffic_app_user
--

ALTER SEQUENCE public.daily_traffic_summary_id_seq OWNED BY public.daily_traffic_summary.id;


--
-- TOC entry 230 (class 1259 OID 16516)
-- Name: traffic_events; Type: TABLE; Schema: public; Owner: traffic_app_user
--

CREATE TABLE public.traffic_events (
    id integer NOT NULL,
    camera_id integer,
    event_time timestamp without time zone NOT NULL,
    event_type character varying(50),
    description text
);


ALTER TABLE public.traffic_events OWNER TO traffic_app_user;

--
-- TOC entry 229 (class 1259 OID 16515)
-- Name: traffic_events_id_seq; Type: SEQUENCE; Schema: public; Owner: traffic_app_user
--

CREATE SEQUENCE public.traffic_events_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.traffic_events_id_seq OWNER TO traffic_app_user;

--
-- TOC entry 4900 (class 0 OID 0)
-- Dependencies: 229
-- Name: traffic_events_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: traffic_app_user
--

ALTER SEQUENCE public.traffic_events_id_seq OWNED BY public.traffic_events.id;


--
-- TOC entry 218 (class 1259 OID 16391)
-- Name: users; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.users (
    id integer NOT NULL,
    username character varying(50) NOT NULL,
    password character varying(100) NOT NULL,
    full_name text,
    role character varying(20) DEFAULT 'user'::character varying,
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP
);


ALTER TABLE public.users OWNER TO postgres;

--
-- TOC entry 217 (class 1259 OID 16390)
-- Name: users_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.users_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.users_id_seq OWNER TO postgres;

--
-- TOC entry 4902 (class 0 OID 0)
-- Dependencies: 217
-- Name: users_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.users_id_seq OWNED BY public.users.id;


--
-- TOC entry 232 (class 1259 OID 16532)
-- Name: vehicle_stats; Type: TABLE; Schema: public; Owner: traffic_app_user
--

CREATE TABLE public.vehicle_stats (
    id integer NOT NULL,
    camera_id integer,
    date date NOT NULL,
    time_slot smallint NOT NULL,
    direction character varying(10),
    vehicle_type character varying(50),
    vehicle_count integer,
    CONSTRAINT vehicle_stats_direction_check CHECK (((direction)::text = ANY ((ARRAY['top'::character varying, 'bottom'::character varying, 'left'::character varying, 'right'::character varying])::text[]))),
    CONSTRAINT vehicle_stats_time_slot_check CHECK (((time_slot >= 0) AND (time_slot <= 47))),
    CONSTRAINT vehicle_stats_vehicle_count_check CHECK ((vehicle_count >= 0)),
    CONSTRAINT vehicle_stats_vehicle_type_check CHECK (((vehicle_type)::text = ANY ((ARRAY['car'::character varying, 'motorbike'::character varying, 'truck'::character varying, 'bus'::character varying, 'bicycle'::character varying])::text[])))
);


ALTER TABLE public.vehicle_stats OWNER TO traffic_app_user;

--
-- TOC entry 231 (class 1259 OID 16531)
-- Name: vehicle_stats_id_seq; Type: SEQUENCE; Schema: public; Owner: traffic_app_user
--

CREATE SEQUENCE public.vehicle_stats_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.vehicle_stats_id_seq OWNER TO traffic_app_user;

--
-- TOC entry 4903 (class 0 OID 0)
-- Dependencies: 231
-- Name: vehicle_stats_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: traffic_app_user
--

ALTER SEQUENCE public.vehicle_stats_id_seq OWNED BY public.vehicle_stats.id;


--
-- TOC entry 4683 (class 2604 OID 16456)
-- Name: areas id; Type: DEFAULT; Schema: public; Owner: traffic_app_user
--

ALTER TABLE ONLY public.areas ALTER COLUMN id SET DEFAULT nextval('public.areas_id_seq'::regclass);


--
-- TOC entry 4682 (class 2604 OID 16440)
-- Name: avg_speeds id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.avg_speeds ALTER COLUMN id SET DEFAULT nextval('public.avg_speeds_id_seq'::regclass);


--
-- TOC entry 4684 (class 2604 OID 16465)
-- Name: camera_area id; Type: DEFAULT; Schema: public; Owner: traffic_app_user
--

ALTER TABLE ONLY public.camera_area ALTER COLUMN id SET DEFAULT nextval('public.camera_area_id_seq'::regclass);


--
-- TOC entry 4680 (class 2604 OID 16407)
-- Name: cameras id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.cameras ALTER COLUMN id SET DEFAULT nextval('public.cameras_id_seq'::regclass);


--
-- TOC entry 4685 (class 2604 OID 16502)
-- Name: daily_traffic_summary id; Type: DEFAULT; Schema: public; Owner: traffic_app_user
--

ALTER TABLE ONLY public.daily_traffic_summary ALTER COLUMN id SET DEFAULT nextval('public.daily_traffic_summary_id_seq'::regclass);


--
-- TOC entry 4686 (class 2604 OID 16519)
-- Name: traffic_events id; Type: DEFAULT; Schema: public; Owner: traffic_app_user
--

ALTER TABLE ONLY public.traffic_events ALTER COLUMN id SET DEFAULT nextval('public.traffic_events_id_seq'::regclass);


--
-- TOC entry 4677 (class 2604 OID 16394)
-- Name: users id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.users ALTER COLUMN id SET DEFAULT nextval('public.users_id_seq'::regclass);


--
-- TOC entry 4687 (class 2604 OID 16535)
-- Name: vehicle_stats id; Type: DEFAULT; Schema: public; Owner: traffic_app_user
--

ALTER TABLE ONLY public.vehicle_stats ALTER COLUMN id SET DEFAULT nextval('public.vehicle_stats_id_seq'::regclass);


--
-- TOC entry 4879 (class 0 OID 16453)
-- Dependencies: 224
-- Data for Name: areas; Type: TABLE DATA; Schema: public; Owner: traffic_app_user
--

COPY public.areas (id, name, description) FROM stdin;
1	Phường Thảo Điền	Quận 2, TP.HCM
2	Phường Tân Phú	TP Thủ Đức, TP.HCM
3	Phường Hiệp Phú	TP Thủ Đức, TP.HCM
4	Phường Linh Trung	TP Thủ Đức, TP.HCM
5	Khu KCNC	TP Thủ Đức, TP.HCM
\.


--
-- TOC entry 4877 (class 0 OID 16437)
-- Dependencies: 222
-- Data for Name: avg_speeds; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.avg_speeds (id, camera_id, date, time_slot, average_speed) FROM stdin;
33	4	2025-06-06	34	62.22
35	2	2025-06-06	34	0
34	6	2025-06-06	34	23.6
32	1	2025-06-06	34	36.03
31	5	2025-06-06	34	36.06
36	3	2025-06-06	34	22.98
73	4	2025-06-10	38	39.5
74	6	2025-06-10	38	21.33
75	5	2025-06-10	38	36.36
76	3	2025-06-10	38	27.13
77	1	2025-06-10	38	40.33
78	2	2025-06-10	38	37.89
80	1	2025-06-10	39	40.93
81	5	2025-06-10	39	34.46
82	6	2025-06-10	39	22.55
83	3	2025-06-10	39	32.17
84	2	2025-06-10	39	36.16
85	4	2025-06-10	40	43.54
86	1	2025-06-10	40	33.27
87	5	2025-06-10	40	35.16
88	6	2025-06-10	40	19.78
89	3	2025-06-10	40	26.41
90	2	2025-06-10	40	33.63
79	4	2025-06-10	39	62.34
94	4	2025-06-12	46	0
96	2	2025-06-12	46	26.89
93	1	2025-06-12	46	0
95	5	2025-06-12	46	38.76
92	3	2025-06-12	46	23.45
91	6	2025-06-12	46	17.27
117	4	2025-06-12	47	0
120	1	2025-06-12	47	63.08
119	3	2025-06-12	47	17.12
115	2	2025-06-12	47	23.42
116	6	2025-06-12	47	19.02
118	5	2025-06-12	47	21.97
134	5	2025-06-13	23	0
133	4	2025-06-13	23	0
135	2	2025-06-13	23	0
136	3	2025-06-13	23	0
137	1	2025-06-13	23	0
138	6	2025-06-13	23	0
139	6	2025-06-13	24	24.04
140	5	2025-06-13	24	34.83
141	2	2025-06-13	24	38.76
142	1	2025-06-13	24	36.44
143	4	2025-06-13	24	34.75
144	3	2025-06-13	24	25.89
\.


--
-- TOC entry 4881 (class 0 OID 16462)
-- Dependencies: 226
-- Data for Name: camera_area; Type: TABLE DATA; Schema: public; Owner: traffic_app_user
--

COPY public.camera_area (id, camera_id, area_id, location_detail) FROM stdin;
2	2	2	Xa lộ Hà Nội – Đường D400
3	3	3	Xa lộ Hà Nội – Đường Lê Văn Việt
4	4	4	Xa lộ Hà Nội – Lên cầu Ngã Tư Thủ Đức
5	5	1	Xa lộ Hà Nội – Thảo Điền
6	6	2	Xa lộ Hà Nội – Đường 120
1	1	5	Xa lộ Hà Nội – Đường D1
\.


--
-- TOC entry 4875 (class 0 OID 16404)
-- Dependencies: 220
-- Data for Name: cameras; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.cameras (id, name, location, installed_at) FROM stdin;
1	Camera 1	Xa lo ha noi - Duong D1	2025-05-26 12:08:44.437418
2	Camera 2	Xa lo ha noi - Duong D400	2025-05-26 12:08:44.437418
3	Camera 3	Xa lo ha noi - Duong Le Van Viet	2025-05-26 12:08:44.437418
4	Camera 4	Xa lo ha noi - Len cau Nga tu thu duc	2025-05-26 12:08:44.437418
5	Camera 5	Xa lo ha noi - Thao Dien	2025-05-26 12:08:44.437418
6	Camera 6	Xa lo ha noi - Duong 120	2025-05-26 12:08:44.437418
\.


--
-- TOC entry 4883 (class 0 OID 16499)
-- Dependencies: 228
-- Data for Name: daily_traffic_summary; Type: TABLE DATA; Schema: public; Owner: traffic_app_user
--

COPY public.daily_traffic_summary (id, camera_id, date, total_vehicle_count, avg_speed, peak_time_slot, direction_with_most_traffic) FROM stdin;
61	4	2025-06-10	44	43.54	40	bottom
65	1	2025-06-10	339	33.27	40	top
63	5	2025-06-10	320	35.16	40	bottom
62	6	2025-06-10	148	19.78	40	bottom
64	3	2025-06-10	471	26.41	40	top
66	2	2025-06-10	186	33.63	40	bottom
21	4	2025-06-06	9	62.22	34	top
23	2	2025-06-06	59	0	34	unknown
22	6	2025-06-06	22	23.6	34	bottom
20	1	2025-06-06	53	36.03	34	top
19	5	2025-06-06	56	36.06	34	bottom
24	3	2025-06-06	71	22.98	34	top
82	4	2025-06-12	20	0	47	unknown
81	1	2025-06-12	97	63.08	47	unknown
80	3	2025-06-12	147	17.12	47	top
84	2	2025-06-12	80	23.42	47	bottom
79	6	2025-06-12	57	19.02	47	unknown
83	5	2025-06-12	131	21.97	47	unknown
126	6	2025-06-13	35	24.04	24	bottom
121	5	2025-06-13	76	34.83	24	bottom
122	2	2025-06-13	67	38.76	24	bottom
124	1	2025-06-13	67	36.44	24	top
123	4	2025-06-13	14	34.75	24	top
125	3	2025-06-13	113	25.89	24	top
\.


--
-- TOC entry 4885 (class 0 OID 16516)
-- Dependencies: 230
-- Data for Name: traffic_events; Type: TABLE DATA; Schema: public; Owner: traffic_app_user
--

COPY public.traffic_events (id, camera_id, event_time, event_type, description) FROM stdin;
1	1	2025-08-12 07:30:00	congestion	Kẹt xe nghiêm trọng vào giờ cao điểm buổi sáng gần Đường D1.
2	2	2025-09-03 18:15:00	accident	Va chạm nhẹ giữa hai ô tô xảy ra tại Đường D400.
3	3	2025-10-05 14:00:00	roadwork	Thi công bảo trì mặt đường tại khu vực Lê Văn Việt – đóng một làn đường.
4	4	2025-08-25 17:45:00	congestion	Ùn tắc giao thông tại Ngã tư Thủ Đức do lượng xe đông vào giờ tan tầm.
5	5	2025-09-20 09:00:00	roadwork	Cải tạo hệ thống thoát nước tại khu vực Thảo Điền – hạn chế giao thông.
6	6	2025-10-10 20:00:00	accident	Tai nạn giao thông giữa hai xe máy tại khu vực Đường 120 – có lực lượng chức năng xử lý.
\.


--
-- TOC entry 4873 (class 0 OID 16391)
-- Dependencies: 218
-- Data for Name: users; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.users (id, username, password, full_name, role, created_at) FROM stdin;
1	admin	$pbkdf2-sha256$29000$OacUAuAcw3iP8f7/f0.pdQ$h/DsXpiAFKb072VfMt2N5QnlxZCLVNF4lIPct8WN0VQ\n	Administrator	admin	2025-05-10 22:26:11.873829
\.


--
-- TOC entry 4887 (class 0 OID 16532)
-- Dependencies: 232
-- Data for Name: vehicle_stats; Type: TABLE DATA; Schema: public; Owner: traffic_app_user
--

COPY public.vehicle_stats (id, camera_id, date, time_slot, direction, vehicle_type, vehicle_count) FROM stdin;
84	4	2025-06-06	34	bottom	car	1
39	4	2025-06-06	34	top	truck	2
77	2	2025-06-06	34	bottom	bus	2
42	2	2025-06-06	34	right	motorbike	7
106	4	2025-06-06	34	top	motorbike	1
55	2	2025-06-06	34	bottom	car	6
114	1	2025-06-06	34	top	bus	3
43	2	2025-06-06	34	bottom	motorbike	43
48	3	2025-06-06	34	bottom	car	1
119	1	2025-06-06	34	bottom	truck	1
54	6	2025-06-06	34	bottom	car	1
40	4	2025-06-06	34	bottom	truck	2
59	2	2025-06-06	34	right	truck	1
60	1	2025-06-06	34	right	truck	1
62	1	2025-06-06	34	left	truck	1
45	3	2025-06-06	34	right	motorbike	7
68	3	2025-06-06	34	bottom	motorbike	4
78	1	2025-06-06	34	left	bus	1
46	3	2025-06-06	34	left	car	9
50	3	2025-06-06	34	top	motorbike	5
49	3	2025-06-06	34	top	car	12
47	3	2025-06-06	34	left	motorbike	7
38	4	2025-06-06	34	top	car	3
41	6	2025-06-06	34	bottom	motorbike	21
56	1	2025-06-06	34	top	truck	8
35	1	2025-06-06	34	top	car	25
107	1	2025-06-06	34	right	car	2
37	1	2025-06-06	34	bottom	car	11
36	5	2025-06-06	34	bottom	motorbike	51
52	5	2025-06-06	34	bottom	car	5
99	3	2025-06-06	34	right	truck	4
44	3	2025-06-06	34	right	car	9
83	3	2025-06-06	34	top	truck	10
82	3	2025-06-06	34	left	truck	3
141	4	2025-06-10	38	top	car	4
142	4	2025-06-10	38	top	truck	2
143	4	2025-06-10	38	top	motorbike	1
144	4	2025-06-10	38	top	bus	2
145	4	2025-06-10	38	bottom	truck	4
146	4	2025-06-10	38	bottom	car	3
147	4	2025-06-10	38	bottom	bus	1
148	4	2025-06-10	38	bottom	motorbike	1
149	3	2025-06-10	38	right	car	10
150	6	2025-06-10	38	bottom	motorbike	43
151	3	2025-06-10	38	right	motorbike	15
152	6	2025-06-10	38	bottom	car	11
153	3	2025-06-10	38	right	truck	6
154	3	2025-06-10	38	right	bus	1
155	3	2025-06-10	38	left	car	15
156	1	2025-06-10	38	top	motorbike	12
157	3	2025-06-10	38	left	motorbike	19
158	3	2025-06-10	38	left	truck	4
159	1	2025-06-10	38	top	car	40
160	3	2025-06-10	38	left	bus	2
161	1	2025-06-10	38	top	truck	8
162	5	2025-06-10	38	bottom	motorbike	92
163	3	2025-06-10	38	top	car	16
164	1	2025-06-10	38	top	bus	6
165	5	2025-06-10	38	bottom	car	18
166	3	2025-06-10	38	top	motorbike	41
167	1	2025-06-10	38	right	car	6
168	5	2025-06-10	38	bottom	bus	2
169	3	2025-06-10	38	top	truck	9
170	1	2025-06-10	38	right	motorbike	2
171	5	2025-06-10	38	bottom	truck	2
172	3	2025-06-10	38	bottom	motorbike	2
173	1	2025-06-10	38	right	truck	2
174	1	2025-06-10	38	bottom	car	13
175	1	2025-06-10	38	bottom	truck	2
176	2	2025-06-10	38	right	motorbike	11
177	2	2025-06-10	38	right	truck	1
178	2	2025-06-10	38	right	car	1
179	2	2025-06-10	38	bottom	motorbike	44
180	2	2025-06-10	38	bottom	car	15
181	2	2025-06-10	38	bottom	bus	2
182	2	2025-06-10	38	bottom	truck	1
183	4	2025-06-10	39	top	bus	1
184	4	2025-06-10	39	top	truck	2
185	4	2025-06-10	39	top	car	1
186	4	2025-06-10	39	bottom	truck	1
187	4	2025-06-10	39	bottom	car	1
188	1	2025-06-10	39	bottom	car	2
189	1	2025-06-10	39	bottom	motorbike	8
190	1	2025-06-10	39	top	car	10
191	1	2025-06-10	39	top	motorbike	19
192	1	2025-06-10	39	right	car	2
193	1	2025-06-10	39	right	motorbike	54
194	1	2025-06-10	39	left	motorbike	19
195	5	2025-06-10	39	bottom	motorbike	61
196	5	2025-06-10	39	bottom	car	16
197	3	2025-06-10	39	top	truck	1
198	3	2025-06-10	39	top	motorbike	43
199	3	2025-06-10	39	top	bus	2
200	6	2025-06-10	39	bottom	car	9
201	3	2025-06-10	39	top	car	2
202	6	2025-06-10	39	bottom	motorbike	23
203	3	2025-06-10	39	left	motorbike	34
204	6	2025-06-10	39	bottom	truck	1
205	3	2025-06-10	39	left	bus	1
206	3	2025-06-10	39	left	car	5
207	3	2025-06-10	39	bottom	motorbike	3
208	3	2025-06-10	39	bottom	car	1
209	3	2025-06-10	39	right	motorbike	28
210	3	2025-06-10	39	right	car	2
211	2	2025-06-10	39	right	motorbike	9
212	2	2025-06-10	39	right	car	3
213	2	2025-06-10	39	right	bus	1
214	2	2025-06-10	39	bottom	motorbike	41
215	2	2025-06-10	39	bottom	car	3
216	2	2025-06-10	39	bottom	truck	3
217	2	2025-06-10	39	bottom	bus	1
218	4	2025-06-10	40	bottom	truck	5
219	4	2025-06-10	40	bottom	car	5
220	4	2025-06-10	40	bottom	bus	1
221	4	2025-06-10	40	bottom	motorbike	1
222	4	2025-06-10	40	top	truck	3
223	4	2025-06-10	40	top	car	3
224	4	2025-06-10	40	top	motorbike	1
225	4	2025-06-10	40	top	bus	1
226	1	2025-06-10	40	right	motorbike	30
227	1	2025-06-10	40	right	car	15
228	1	2025-06-10	40	right	truck	2
229	1	2025-06-10	40	left	motorbike	10
230	1	2025-06-10	40	left	truck	1
231	1	2025-06-10	40	bottom	motorbike	2
232	1	2025-06-10	40	bottom	car	9
233	1	2025-06-10	40	top	motorbike	16
234	1	2025-06-10	40	top	car	41
235	1	2025-06-10	40	top	truck	2
236	1	2025-06-10	40	top	bus	6
237	5	2025-06-10	40	bottom	motorbike	97
238	5	2025-06-10	40	bottom	car	28
239	5	2025-06-10	40	bottom	bus	2
240	5	2025-06-10	40	bottom	truck	2
241	6	2025-06-10	40	bottom	motorbike	41
242	6	2025-06-10	40	bottom	car	19
243	3	2025-06-10	40	bottom	motorbike	8
244	6	2025-06-10	40	bottom	truck	1
245	3	2025-06-10	40	bottom	car	1
246	3	2025-06-10	40	left	motorbike	28
247	3	2025-06-10	40	left	car	22
248	3	2025-06-10	40	left	truck	2
249	3	2025-06-10	40	top	motorbike	57
250	3	2025-06-10	40	top	car	28
251	3	2025-06-10	40	top	truck	6
252	3	2025-06-10	40	right	motorbike	38
253	3	2025-06-10	40	right	car	17
254	3	2025-06-10	40	right	truck	2
255	2	2025-06-10	40	bottom	motorbike	26
256	2	2025-06-10	40	bottom	bus	2
257	2	2025-06-10	40	bottom	car	8
258	2	2025-06-10	40	right	motorbike	10
259	2	2025-06-10	40	right	car	3
260	2	2025-06-10	40	right	truck	1
261	3	2025-06-12	46	right	car	10
266	3	2025-06-12	46	left	car	14
276	1	2025-06-12	46	top	bus	6
277	1	2025-06-12	46	right	car	5
278	1	2025-06-12	46	right	motorbike	2
282	4	2025-06-12	46	top	truck	2
283	4	2025-06-12	46	top	motorbike	1
286	4	2025-06-12	46	bottom	bus	1
288	2	2025-06-12	46	right	truck	1
293	2	2025-06-12	46	bottom	bus	2
294	2	2025-06-12	46	bottom	truck	1
274	1	2025-06-12	46	top	car	39
296	4	2025-06-12	46	top	bus	1
289	5	2025-06-12	46	bottom	motorbike	79
281	4	2025-06-12	46	top	car	4
299	4	2025-06-12	46	bottom	motorbike	1
292	5	2025-06-12	46	bottom	car	17
284	4	2025-06-12	46	bottom	truck	4
279	1	2025-06-12	46	bottom	car	13
269	3	2025-06-12	46	top	car	16
280	1	2025-06-12	46	bottom	truck	2
267	3	2025-06-12	46	left	motorbike	19
291	2	2025-06-12	46	bottom	car	15
308	3	2025-06-12	46	left	bus	2
307	1	2025-06-12	46	right	truck	2
347	5	2025-06-12	47	bottom	car	5
270	3	2025-06-12	46	top	motorbike	36
311	2	2025-06-12	46	right	car	1
344	5	2025-06-12	47	bottom	motorbike	26
313	5	2025-06-12	46	bottom	bus	2
273	1	2025-06-12	46	top	motorbike	12
275	1	2025-06-12	46	top	truck	5
271	3	2025-06-12	46	top	truck	9
317	5	2025-06-12	46	bottom	truck	2
272	3	2025-06-12	46	bottom	motorbike	2
264	3	2025-06-12	46	right	motorbike	14
265	3	2025-06-12	46	right	truck	6
321	3	2025-06-12	46	right	bus	1
268	3	2025-06-12	46	left	truck	3
285	4	2025-06-12	46	bottom	car	3
287	2	2025-06-12	46	right	motorbike	9
290	2	2025-06-12	46	bottom	motorbike	44
262	6	2025-06-12	46	bottom	motorbike	41
263	6	2025-06-12	46	bottom	car	9
340	2	2025-06-12	47	right	motorbike	4
341	6	2025-06-12	47	bottom	motorbike	3
342	6	2025-06-12	47	bottom	car	4
343	3	2025-06-12	47	left	truck	1
345	4	2025-06-12	47	top	bus	2
346	3	2025-06-12	47	left	car	1
348	3	2025-06-12	47	left	motorbike	1
350	3	2025-06-12	47	top	truck	1
351	3	2025-06-12	47	right	motorbike	1
352	1	2025-06-12	47	right	car	2
354	1	2025-06-12	47	top	truck	3
356	4	2025-06-12	47	bottom	truck	1
358	2	2025-06-12	47	right	car	2
353	1	2025-06-12	47	top	car	4
355	1	2025-06-12	47	bottom	car	2
349	3	2025-06-12	47	top	motorbike	10
363	2	2025-06-12	47	bottom	motorbike	1
365	2	2025-06-13	24	right	motorbike	7
366	1	2025-06-13	24	top	motorbike	7
367	2	2025-06-13	24	right	truck	1
368	1	2025-06-13	24	top	car	34
369	1	2025-06-13	24	top	truck	1
370	2	2025-06-13	24	bottom	motorbike	43
371	4	2025-06-13	24	top	car	3
372	6	2025-06-13	24	bottom	motorbike	31
373	5	2025-06-13	24	bottom	motorbike	63
374	1	2025-06-13	24	top	bus	6
375	2	2025-06-13	24	bottom	car	13
376	4	2025-06-13	24	top	truck	2
378	2	2025-06-13	24	bottom	bus	2
377	6	2025-06-13	24	bottom	car	4
379	5	2025-06-13	24	bottom	car	11
380	1	2025-06-13	24	right	car	5
381	2	2025-06-13	24	bottom	truck	1
384	1	2025-06-13	24	right	motorbike	2
383	4	2025-06-13	24	top	motorbike	1
382	5	2025-06-13	24	bottom	bus	2
385	1	2025-06-13	24	bottom	car	11
386	4	2025-06-13	24	top	bus	1
387	1	2025-06-13	24	bottom	truck	1
388	4	2025-06-13	24	bottom	truck	4
389	4	2025-06-13	24	bottom	car	1
390	4	2025-06-13	24	bottom	bus	1
391	4	2025-06-13	24	bottom	motorbike	1
392	3	2025-06-13	24	right	car	10
393	3	2025-06-13	24	right	motorbike	11
394	3	2025-06-13	24	right	truck	4
395	3	2025-06-13	24	left	car	14
396	3	2025-06-13	24	left	motorbike	19
397	3	2025-06-13	24	left	truck	2
398	3	2025-06-13	24	top	car	12
399	3	2025-06-13	24	top	motorbike	32
400	3	2025-06-13	24	top	truck	7
401	3	2025-06-13	24	bottom	motorbike	2
\.


--
-- TOC entry 4904 (class 0 OID 0)
-- Dependencies: 223
-- Name: areas_id_seq; Type: SEQUENCE SET; Schema: public; Owner: traffic_app_user
--

SELECT pg_catalog.setval('public.areas_id_seq', 5, true);


--
-- TOC entry 4905 (class 0 OID 0)
-- Dependencies: 221
-- Name: avg_speeds_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.avg_speeds_id_seq', 144, true);


--
-- TOC entry 4906 (class 0 OID 0)
-- Dependencies: 225
-- Name: camera_area_id_seq; Type: SEQUENCE SET; Schema: public; Owner: traffic_app_user
--

SELECT pg_catalog.setval('public.camera_area_id_seq', 6, true);


--
-- TOC entry 4907 (class 0 OID 0)
-- Dependencies: 219
-- Name: cameras_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.cameras_id_seq', 1, false);


--
-- TOC entry 4908 (class 0 OID 0)
-- Dependencies: 227
-- Name: daily_traffic_summary_id_seq; Type: SEQUENCE SET; Schema: public; Owner: traffic_app_user
--

SELECT pg_catalog.setval('public.daily_traffic_summary_id_seq', 132, true);


--
-- TOC entry 4909 (class 0 OID 0)
-- Dependencies: 229
-- Name: traffic_events_id_seq; Type: SEQUENCE SET; Schema: public; Owner: traffic_app_user
--

SELECT pg_catalog.setval('public.traffic_events_id_seq', 6, true);


--
-- TOC entry 4910 (class 0 OID 0)
-- Dependencies: 217
-- Name: users_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.users_id_seq', 1, true);


--
-- TOC entry 4911 (class 0 OID 0)
-- Dependencies: 231
-- Name: vehicle_stats_id_seq; Type: SEQUENCE SET; Schema: public; Owner: traffic_app_user
--

SELECT pg_catalog.setval('public.vehicle_stats_id_seq', 401, true);


--
-- TOC entry 4708 (class 2606 OID 16460)
-- Name: areas areas_pkey; Type: CONSTRAINT; Schema: public; Owner: traffic_app_user
--

ALTER TABLE ONLY public.areas
    ADD CONSTRAINT areas_pkey PRIMARY KEY (id);


--
-- TOC entry 4704 (class 2606 OID 16446)
-- Name: avg_speeds avg_speeds_camera_id_date_time_slot_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.avg_speeds
    ADD CONSTRAINT avg_speeds_camera_id_date_time_slot_key UNIQUE (camera_id, date, time_slot);


--
-- TOC entry 4706 (class 2606 OID 16444)
-- Name: avg_speeds avg_speeds_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.avg_speeds
    ADD CONSTRAINT avg_speeds_pkey PRIMARY KEY (id);


--
-- TOC entry 4710 (class 2606 OID 16469)
-- Name: camera_area camera_area_pkey; Type: CONSTRAINT; Schema: public; Owner: traffic_app_user
--

ALTER TABLE ONLY public.camera_area
    ADD CONSTRAINT camera_area_pkey PRIMARY KEY (id);


--
-- TOC entry 4702 (class 2606 OID 16412)
-- Name: cameras cameras_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.cameras
    ADD CONSTRAINT cameras_pkey PRIMARY KEY (id);


--
-- TOC entry 4712 (class 2606 OID 16509)
-- Name: daily_traffic_summary daily_traffic_summary_camera_id_date_key; Type: CONSTRAINT; Schema: public; Owner: traffic_app_user
--

ALTER TABLE ONLY public.daily_traffic_summary
    ADD CONSTRAINT daily_traffic_summary_camera_id_date_key UNIQUE (camera_id, date);


--
-- TOC entry 4714 (class 2606 OID 16507)
-- Name: daily_traffic_summary daily_traffic_summary_pkey; Type: CONSTRAINT; Schema: public; Owner: traffic_app_user
--

ALTER TABLE ONLY public.daily_traffic_summary
    ADD CONSTRAINT daily_traffic_summary_pkey PRIMARY KEY (id);


--
-- TOC entry 4716 (class 2606 OID 16523)
-- Name: traffic_events traffic_events_pkey; Type: CONSTRAINT; Schema: public; Owner: traffic_app_user
--

ALTER TABLE ONLY public.traffic_events
    ADD CONSTRAINT traffic_events_pkey PRIMARY KEY (id);


--
-- TOC entry 4698 (class 2606 OID 16400)
-- Name: users users_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.users
    ADD CONSTRAINT users_pkey PRIMARY KEY (id);


--
-- TOC entry 4700 (class 2606 OID 16402)
-- Name: users users_username_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.users
    ADD CONSTRAINT users_username_key UNIQUE (username);


--
-- TOC entry 4718 (class 2606 OID 16543)
-- Name: vehicle_stats vehicle_stats_camera_id_date_time_slot_direction_vehicle_ty_key; Type: CONSTRAINT; Schema: public; Owner: traffic_app_user
--

ALTER TABLE ONLY public.vehicle_stats
    ADD CONSTRAINT vehicle_stats_camera_id_date_time_slot_direction_vehicle_ty_key UNIQUE (camera_id, date, time_slot, direction, vehicle_type);


--
-- TOC entry 4720 (class 2606 OID 16541)
-- Name: vehicle_stats vehicle_stats_pkey; Type: CONSTRAINT; Schema: public; Owner: traffic_app_user
--

ALTER TABLE ONLY public.vehicle_stats
    ADD CONSTRAINT vehicle_stats_pkey PRIMARY KEY (id);


--
-- TOC entry 4721 (class 2606 OID 16447)
-- Name: avg_speeds avg_speeds_camera_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.avg_speeds
    ADD CONSTRAINT avg_speeds_camera_id_fkey FOREIGN KEY (camera_id) REFERENCES public.cameras(id) ON DELETE CASCADE;


--
-- TOC entry 4722 (class 2606 OID 16475)
-- Name: camera_area camera_area_area_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: traffic_app_user
--

ALTER TABLE ONLY public.camera_area
    ADD CONSTRAINT camera_area_area_id_fkey FOREIGN KEY (area_id) REFERENCES public.areas(id) ON DELETE SET NULL;


--
-- TOC entry 4723 (class 2606 OID 16470)
-- Name: camera_area camera_area_camera_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: traffic_app_user
--

ALTER TABLE ONLY public.camera_area
    ADD CONSTRAINT camera_area_camera_id_fkey FOREIGN KEY (camera_id) REFERENCES public.cameras(id) ON DELETE CASCADE;


--
-- TOC entry 4724 (class 2606 OID 16510)
-- Name: daily_traffic_summary daily_traffic_summary_camera_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: traffic_app_user
--

ALTER TABLE ONLY public.daily_traffic_summary
    ADD CONSTRAINT daily_traffic_summary_camera_id_fkey FOREIGN KEY (camera_id) REFERENCES public.cameras(id) ON DELETE CASCADE;


--
-- TOC entry 4725 (class 2606 OID 16524)
-- Name: traffic_events traffic_events_camera_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: traffic_app_user
--

ALTER TABLE ONLY public.traffic_events
    ADD CONSTRAINT traffic_events_camera_id_fkey FOREIGN KEY (camera_id) REFERENCES public.cameras(id) ON DELETE CASCADE;


--
-- TOC entry 4726 (class 2606 OID 16544)
-- Name: vehicle_stats vehicle_stats_camera_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: traffic_app_user
--

ALTER TABLE ONLY public.vehicle_stats
    ADD CONSTRAINT vehicle_stats_camera_id_fkey FOREIGN KEY (camera_id) REFERENCES public.cameras(id) ON DELETE CASCADE;


--
-- TOC entry 4894 (class 0 OID 0)
-- Dependencies: 222
-- Name: TABLE avg_speeds; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON TABLE public.avg_speeds TO traffic_app_user;


--
-- TOC entry 4897 (class 0 OID 0)
-- Dependencies: 220
-- Name: TABLE cameras; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON TABLE public.cameras TO traffic_app_user;


--
-- TOC entry 4901 (class 0 OID 0)
-- Dependencies: 218
-- Name: TABLE users; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON TABLE public.users TO traffic_app_user;


--
-- TOC entry 2079 (class 826 OID 16389)
-- Name: DEFAULT PRIVILEGES FOR TABLES; Type: DEFAULT ACL; Schema: -; Owner: postgres
--

ALTER DEFAULT PRIVILEGES FOR ROLE postgres GRANT ALL ON TABLES TO traffic_app_user;


-- Completed on 2025-06-13 12:17:33

--
-- PostgreSQL database dump complete
--

