void trainNetwork(Network* network){
//TODO: we should implement the mpi init in here
    // printf("Inside");
    MPI_Init(NULL,NULL);
    int procs, myid;
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &procs);

    if(myid==0){
    FILE *imageFile;
    FILE *labelFile;
    ImageFileHeader imageFileHeader;
    imageFile = openImageFile(TRAINING_SET_IMAGE_FILE_NAME, &imageFileHeader);
    // printf("%s",TRAINING_SET_LABEL_FILE_NAME);
    labelFile = openLabelFile(TRAINING_SET_LABEL_FILE_NAME);
    }

    for(int i=0; i<imageFileHeader.maxImages; i++){
        Image img;
        /////// pack
    int imgType, imgHeight, imgWidth, numLines;
    void **pixelMat; // store the pixel lines assigned to the current process
    bounds jobBounds;
    MPI_Datatype pixelDataType;

    // create MPI type for color pixel struct
    int blocklengths[3] = {1, 1, 1};
    MPI_Aint offsets[3] = {0, 1, 2};
    MPI_Datatype types[3] = {MPI_UNSIGNED_CHAR, MPI_UNSIGNED_CHAR, MPI_UNSIGNED_CHAR};
    MPI_Datatype MPI_3UNSIGNED_CHAR;
    MPI_Type_create_struct(3, blocklengths, offsets, types, &MPI_3UNSIGNED_CHAR);
    MPI_Type_commit(&MPI_3UNSIGNED_CHAR);


    ////// end pack
        getImage(imageFile, &img);
        uint8_t label = getLabel(labelFile);

        feedForward(network, &img);
        
        backPropagate(network, label);
    }
    MPI_Finalize();
}
